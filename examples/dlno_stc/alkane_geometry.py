"""Extend the supplied C20 all-trans chain by repeating its central CH2 pair.

C20 is copied exactly. Longer chains preserve both original end regions and
insert an even number of CH2 units at the central bond. Coordinates are in
angstrom; these are benchmark geometries, without a new geometry optimization.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path


def read_xyz(path):
    lines = Path(path).read_text().splitlines()
    atoms = [(parts[0], tuple(map(float, parts[1:4])))
             for line in lines[2:] if (parts := line.split())]
    if len(atoms) != int(lines[0]) or any(len(xyz) != 3 for _, xyz in atoms):
        raise ValueError('invalid XYZ atom count or coordinates')
    return atoms


def distance(a, b):
    return math.sqrt(sum((x-y)**2 for x, y in zip(a, b)))


def grouped_chain(atoms):
    carbons = sorted((xyz for symbol, xyz in atoms if symbol == 'C'), key=lambda p: p[0])
    groups = [[('C', xyz)] for xyz in carbons]
    for symbol, xyz in atoms:
        if symbol == 'H':
            owner = min(range(len(carbons)), key=lambda i: distance(xyz, carbons[i]))
            groups[owner].append(('H', xyz))
        elif symbol != 'C':
            raise ValueError('alkanes must contain only C and H')
    return groups


def validate(atoms, carbons):
    if len(atoms) != 3*carbons+2 or sum(s == 'C' for s, _ in atoms) != carbons:
        raise ValueError('incorrect alkane formula')
    if not all(math.isfinite(v) for _, xyz in atoms for v in xyz):
        raise ValueError('nonfinite coordinate')
    groups = grouped_chain(atoms)
    cc = [distance(a[0][1], b[0][1]) for a, b in zip(groups, groups[1:])]
    ch = [distance(g[0][1], xyz) for g in groups for _, xyz in g[1:]]
    if not all(1.45 < x < 1.65 for x in cc) or not all(1.0 < x < 1.2 for x in ch):
        raise ValueError('invalid C-C or C-H bond length')
    if [len(g)-1 for g in groups] != [3] + [2]*(carbons-2) + [3]:
        raise ValueError('invalid hydrogen connectivity')
    minimum = min(distance(a[1], b[1]) for i, a in enumerate(atoms) for b in atoms[i+1:])
    if minimum < .9:
        raise ValueError('overlapping atoms')
    return dict(carbons=carbons, hydrogens=2*carbons+2, atoms=len(atoms),
                cc_bond_range_angstrom=[min(cc), max(cc)],
                ch_bond_range_angstrom=[min(ch), max(ch)],
                minimum_atom_separation_angstrom=minimum)


def write_chain(template, carbons, output):
    template, output = Path(template), Path(output)
    if carbons < 20 or (carbons-20) % 2:
        raise ValueError('extension requires an even carbon count of at least 20')
    original = read_xyz(template)
    validate(original, 20)
    if carbons == 20:
        atoms, payload = original, template.read_bytes()
    else:
        groups = grouped_chain(original)
        repeat = tuple(b-a for a, b in zip(groups[10][0][1], groups[12][0][1]))
        pairs = (carbons-20)//2
        def translated(group, count):
            return [(symbol, tuple(x+count*t for x, t in zip(xyz, repeat)))
                    for symbol, xyz in group]
        extended = groups[:10]
        for count in range(pairs):
            extended += [translated(g, count) for g in groups[10:12]]
        extended += [translated(g, pairs) for g in groups[10:]]
        atoms = [atom for group in extended for atom in group]
        center = tuple(sum(xyz[k] for symbol, xyz in atoms if symbol == 'C')/carbons
                       for k in range(3))
        atoms = [(symbol, tuple(x-c for x, c in zip(xyz, center))) for symbol, xyz in atoms]
        lines = [str(len(atoms)), f'C{carbons}H{2*carbons+2}; central CH2-pair extension of supplied C20']
        lines += [f'{s} {p[0]:.12f} {p[1]:.12f} {p[2]:.12f}' for s, p in atoms]
        payload = ('\n'.join(lines)+'\n').encode()
    report = validate(atoms, carbons)
    report.update(template=str(template.resolve()), template_sha256=hashlib.sha256(template.read_bytes()).hexdigest(),
                  geometry_sha256=hashlib.sha256(payload).hexdigest(), method='central CH2-pair extension; C20 copied exactly')
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and output.read_bytes() != payload:
        raise ValueError(f'existing geometry differs: {output}')
    output.write_bytes(payload)
    output.with_suffix('.json').write_text(json.dumps(report, indent=2)+'\n')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--template', type=Path, required=True)
    parser.add_argument('--carbons', type=int, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(write_chain(args.template, args.carbons, args.output), indent=2))
