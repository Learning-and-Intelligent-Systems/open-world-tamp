#ifndef PDBS_PATTERN_COLLECTION_GENERATOR_SINGLE_CEGAR_H
#define PDBS_PATTERN_COLLECTION_GENERATOR_SINGLE_CEGAR_H

#include "pattern_generator.h"

namespace utils {
class RandomNumberGenerator;
enum class Verbosity;
}

namespace pdbs {
/*
  This pattern collection generator uses the CEGAR algorithm to compute a
  disjoint pattern collection for the given task. See cegar.h for more details.
*/
class PatternCollectionGeneratorSingleCegar : public PatternCollectionGenerator {
    const int max_pdb_size;
    const int max_collection_size;
    const bool use_wildcard_plans;
    const double max_time;
    const utils::Verbosity verbosity;
    std::shared_ptr<utils::RandomNumberGenerator> rng;
public:
    explicit PatternCollectionGeneratorSingleCegar(const options::Options &opts);

    virtual PatternCollectionInformation generate(
        const std::shared_ptr<AbstractTask> &task) override;
};
}

#endif
