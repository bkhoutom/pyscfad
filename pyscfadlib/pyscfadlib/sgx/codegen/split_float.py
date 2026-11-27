"""Split selected straight-line generated kernels into bounded SIMD helpers.

This deliberately accepts only the checked-in VD/VRR generator syntax. Each
output keeps its original expression and primitive accumulation order. Helpers
recompute shared intermediates when needed instead of materializing a large
temporary tensor. Fail on unfamiliar syntax rather than silently dropping work.
"""
import re

MAX_DECLARATIONS = 768
MAX_OUTPUTS = 112
_IDENTIFIERS = re.compile(r'\b[A-Za-z_]\w*\b')
_DECLARATION = re.compile(
    r'^\s*(?:const )?(?:float|T)\s*(?:\*\s*(?:__restrict\s*)?)?'
    r'([A-Za-z_]\w*)\s*=\s*(.*);\s*$')
_OUTPUT = re.compile(r'^\s*(?:out\[[^]]+\]|integral_\d+\[g\])\s*\+?=.+;\s*$')


def closing_brace(text, start):
    depth = 0
    for index in range(start, len(text)):
        if text[index] == '{':
            depth += 1
        elif text[index] == '}':
            depth -= 1
            if not depth:
                return index
    raise ValueError('Unclosed generated function or loop')


def split_kernel(text, name):
    # Keep offsets stable while excluding parameter comments from the parser.
    syntax = re.sub(r'//[^\n]*', lambda match: ' ' * len(match[0]), text)
    function = re.search(r'(template <typename T>\s*)?void ' + name
                         + r'\((.*?)\)\s*\{', syntax, re.DOTALL)
    if function is None:
        raise ValueError(f'Missing generated function {name}')
    template = 'template <typename T>\n' if function[1] else ''
    arguments = re.sub(r'//[^\n]*', '', function[2])
    arguments = ', '.join(arg.strip() for arg in arguments.split(','))
    names = [re.search(r'(\w+)$', arg)[1] for arg in arguments.split(',')]
    start = function.end() - 1
    end = closing_brace(text, start)
    body = text[start + 1:end]
    loops = list(re.finditer(
        r'#pragma omp simd\s*\n\s*for \(int g\s*=\s*0; g\s*<\s*'
        r'(ng|gb_size); \+\+g\)\s*\{', body))
    if not loops:
        raise ValueError(f'No SIMD loops in {name}')

    outer = {}
    for line in body[:loops[0].start()].splitlines():
        match = _DECLARATION.fullmatch(line)
        if match:
            outer[match[1]] = (line.strip(), match[2])

    helpers, replacements, sizes = [], [], []
    output_count = 0
    pair_count = None
    if name.startswith('VD_'):
        offsets = [int(value) for value in re.findall(r'out\[(\d+)\*ng\+g\]', body)]
        pair_count = (max(offsets) + 1) // 7
    for loop in loops:
        loop_end = closing_brace(body, loop.end() - 1)
        inner, outputs = {}, []
        for line in body[loop.end():loop_end].splitlines():
            if not line.strip():
                continue
            match = _DECLARATION.fullmatch(line)
            if match:
                if outputs:
                    raise ValueError('Expected declarations before outputs')
                inner[match[1]] = (line.strip(), match[2])
            elif _OUTPUT.fullmatch(line):
                outputs.append(line.strip())
            else:
                raise ValueError(f'Unexpected statement in {name}: {line}')
        if pair_count:
            # Keep the value and six derivatives of each AO pair together so
            # helpers retain their shared recurrence intermediates.
            outputs.sort(key=lambda line: int(re.search(r'out\[(\d+)\*', line)[1]) % pair_count)
        declarations = dict(outer, **inner)

        def dependencies(statement):
            pending = list(_IDENTIFIERS.findall(statement))
            needed = set()
            while pending:
                key = pending.pop()
                if key in declarations and key not in needed:
                    needed.add(key)
                    pending.extend(_IDENTIFIERS.findall(declarations[key][1]))
            return needed

        chunks, selected, needed = [], [], set()
        for output in outputs:
            deps = dependencies(output)
            if selected and (len(selected) >= MAX_OUTPUTS
                             or len(needed | deps) > MAX_DECLARATIONS):
                chunks.append((selected, needed))
                selected, needed = [], set()
            if len(deps) > MAX_DECLARATIONS:
                raise ValueError(f'One output exceeds the helper bound in {name}')
            selected.append(output)
            needed |= deps
        if selected:
            chunks.append((selected, needed))
        if not chunks:
            raise ValueError(f'No output assignments in {name}')

        calls = []
        for selected, needed in chunks:
            helper = f'{name}_chunk_{len(helpers)}'
            code = [template + f'static PYSCFAD_SGX_NOINLINE void {helper}('
                    + arguments + ', int prim) {']
            code += ['    ' + value[0] for key, value in outer.items() if key in needed]
            code += ['    #pragma omp simd', f'    for (int g=0; g<{loop[1]}; ++g) {{']
            code += ['        ' + value[0] for key, value in inner.items() if key in needed]
            code += ['        ' + output for output in selected]
            code += ['    }', '}']
            helpers.append('\n'.join(code))
            calls.append(helper + ('<T>' if template else '')
                         + '(' + ', '.join(names + ['prim']) + ');')
            sizes.append(len(needed))
            output_count += len(selected)
        replacements.append((loop.start(), loop_end + 1, '\n        '.join(calls)))

    for left, right, replacement in reversed(replacements):
        body = body[:left] + replacement + body[right:]
    # These values now belong to the helpers, not the small public wrapper.
    body = '\n'.join(line for line in body.split('\n')
                     if not _DECLARATION.fullmatch(line))
    attribute = ('#if defined(__GNUC__) || defined(__clang__)\n'
                 '#define PYSCFAD_SGX_NOINLINE __attribute__((noinline))\n'
                 '#elif defined(_MSC_VER)\n'
                 '#define PYSCFAD_SGX_NOINLINE __declspec(noinline)\n'
                 '#else\n#define PYSCFAD_SGX_NOINLINE\n#endif\n')
    result = (text[:function.start()] + attribute + '\n\n'.join(helpers)
              + '\n#undef PYSCFAD_SGX_NOINLINE\n\n'
              + text[function.start():start + 1] + body + text[end:])
    result = '\n'.join(line.rstrip() for line in result.split('\n'))
    return result, dict(helpers=len(helpers), max_declarations=max(sizes),
                        output_assignments=output_count,
                        declaration_limit=MAX_DECLARATIONS, output_limit=MAX_OUTPUTS)
