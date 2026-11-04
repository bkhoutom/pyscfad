#ifndef PYSCFAD_TRIPLES_PREPARED_H
#define PYSCFAD_TRIPLES_PREPARED_H
#include <array>
#include <vector>
#include <cstddef>

namespace libgmbpt { namespace qchem {
// Shared contract between the stable dresser and numerator kernel. Each
// internal contracted leg is undressed. Projected legs use U^T or U^T O^2.
enum triples_kind { T1, T2, GV, GO, GG, FOV, T2V, T2O, NTYPES };
struct triples_tensor_spec {
    size_t kind, source, rank;
    std::array<size_t,4> dims;
    // 0 identity, 1 O, 2 V, 3 U^T, 4 U^T O^2.
    std::array<size_t,4> transform;
};
struct triples_layout {
    std::vector<triples_tensor_spec> tensors;
    size_t projected[2][NTYPES][2];
    triples_layout(size_t no,size_t nv,size_t ns) {
        tensors={
            {T1,0,2,{no,nv,1,1},{1,2,0,0}},
            {T2,1,4,{no,no,nv,nv},{1,1,2,2}},
            {GV,2,4,{no,nv,nv,nv},{1,2,2,0}},
            {GO,3,4,{no,nv,no,no},{1,2,1,0}},
            {GG,4,4,{no,nv,no,nv},{1,2,1,2}},
            {FOV,5,2,{no,nv,1,1},{1,2,0,0}},
            {T2V,1,4,{no,no,nv,nv},{1,1,2,0}},
            {T2O,1,4,{no,no,nv,nv},{0,1,2,2}}};
        for(size_t side=0;side<2;++side) for(size_t kind=0;kind<NTYPES;++kind)
            for(size_t slot=0;slot<2;++slot) {
                projected[side][kind][slot]=kind;
                if(side==0 && (kind==T1 || kind==T2 || kind==GG || kind==FOV)) continue;
                if(kind==T2O && slot!=1) continue;
                const bool two=(kind==T2 || kind==T2V || kind==T2O || kind==GO || kind==GG);
                if(slot && !two) continue;
                const size_t axis=slot ? ((kind==GO || kind==GG) ? 2 : 1) : 0;
                auto spec=tensors[kind];
                spec.dims[axis]=ns;
                spec.transform[axis]=side ? 4 : 3;
                projected[side][kind][slot]=tensors.size();
                tensors.push_back(spec);
            }
    }
};
inline bool triples_doubles_kind(size_t kind) {
    return kind==T2 || kind==T2V || kind==T2O;
}
}}
#endif
