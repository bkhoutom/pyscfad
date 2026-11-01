#include "stc_mp2_helpers.h"
#include <iostream>
#include <limits>

using namespace pyscfad::dlno_stc;

int main() {
    try {
        Controls fixed;
        Moments zero;
        if (production_count(fixed,zero,0.,0.) != 4096)
            throw std::runtime_error("default fixed-count sampling changed");
        Controls controls;
        controls.adaptive = true;
        controls.roots = {0.};
        controls.energy_tolerance = .5;
        Moments pilot;
        pilot.count = 2;
        pilot.m2 = 16000000.;
        // One residual: v=16M, c=1, sigma=.5 requires v/sigma^2=64M.
        if (production_count(controls,pilot,1.,4000.) != 64000000)
            throw std::runtime_error("uncapped allocation did not request 64 million draws");

        controls.max_production_samples = 12345;
        if (production_count(controls,pilot,1.,4000.) != 12345)
            throw std::runtime_error("explicit production limit was not respected");

        controls.max_production_samples = 0;
        controls.energy_tolerance = 1e200;
        if (production_count(controls,pilot,1.,4000.) != 10000)
            throw std::runtime_error("loose tolerance did not use the adaptive minimum");
        controls.energy_tolerance = 1e-300;
        bool overflow = false;
        try { production_count(controls,pilot,1.,4000.); }
        catch (const std::overflow_error&) { overflow = true; }
        if (!overflow) throw std::runtime_error("unrepresentable sampling budget did not fail");

        controls.max_production_samples = 12345;
        if (production_count(controls,pilot,1.,4000.) != 12345)
            throw std::runtime_error("explicit limit did not handle overflowing request");
        controls.max_production_samples = 0;
        controls.energy_tolerance = 1e-10; // Finite request, beyond size_t.
        overflow = false;
        try { production_count(controls,pilot,1.,4000.); }
        catch (const std::overflow_error&) { overflow = true; }
        if (!overflow) throw std::runtime_error("out-of-range finite draw count did not fail");

        pilot.m2 = 0.;
        controls.energy_tolerance = 1e-300;
        controls.min_production_samples = 10000;
        if (production_count(controls,pilot,1.,0.) != 10000)
            throw std::runtime_error("zero-variance pilot lost the production minimum");

        controls.adaptive = false;
        controls.production_samples = 17;
        controls.min_production_samples = 2;
        if (production_count(controls,pilot,0.,0.) != 17)
            throw std::runtime_error("fixed-count sampling changed");

        const auto draws = draw_batched(8195,7,false,{},
            [](std::mt19937_64&,std::vector<Update>&) { return 2.; });
        if (draws.count != 8195 || draws.mean != 2. || draws.variance() != 0.)
            throw std::runtime_error("partial final sampling wave lost draws");
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
    std::cout << "Sampling allocation checks passed\n";
}
