//
// Created by nkk on 2025/10/27.
//

#ifndef TFFINFER_MODELFACTORY_H
#define TFFINFER_MODELFACTORY_H
#include <vector>
#include <memory>
#include <functional>
namespace tff::factory {
    class ModelRegistry {
    public:
        static ModelRegistry& get() {
            static ModelRegistry instance;
            return instance;
        }
        void register_detector(std::unique_ptr<ModelDetector> detector) {
            detectors_.push_back(std::move(detector));
        }

        std::unique_ptr<ModelLoader> find_loader(const ModelConfig & config) const {
            const ModelDetector* best_match = nullptr;
            int best_priority = -1;

            for (auto & detector : detectors_) {
                if (detector->matches(config)) {
                    int prio = detector->priority();
                    if (prio > best_priority) {
                        best_priority = prio;
                        best_match = detector.get();
                    }
                }
            }

            if (best_match) {
                return best_match->create_loader();
            }

            return nullptr;
        }

    private:
        std::vector<std::unique_ptr<ModelDetector>> detectors_;
    };
}
#endif //TFFINFER_MODELFACTORY_H