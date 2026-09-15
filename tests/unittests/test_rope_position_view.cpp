#include "engine/framework/core/backend.h"
#include "engine/framework/modules/positional_modules.h"
#include "engine/framework/modules/structural_modules.h"

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

// Reproduce the second CFG row: one int32 position viewed at byte offset 4.
// Compare with an independently allocated position, so a lost view offset
// cannot make the test pass just by avoiding the Vulkan assertion.
std::vector<float> run(engine::core::BackendType type, bool sliced) {
    using namespace engine;
    auto backend = core::init_backend({type, 0, 2});
    if (!backend) throw std::runtime_error("backend unavailable");
    auto * ggml = ggml_init({4 * 1024 * 1024, nullptr, true});
    if (!ggml) throw std::runtime_error("graph context unavailable");
    core::ModuleBuildContext ctx{ggml, "rope_position_view", type};
    auto input = core::make_tensor(ctx, GGML_TYPE_F32, core::TensorShape::from_dims({1, 1, 1, 64}));
    auto positions = core::make_tensor(ctx, GGML_TYPE_I32, core::TensorShape::from_dims({sliced ? 2 : 1}));
    auto position = sliced ? modules::SliceModule({0, 1, 1}).build(ctx, positions) : positions;
    auto output = modules::RoPEModule({64, GGML_ROPE_TYPE_NEOX, 10000.0f}).build(ctx, input, position);
    auto * graph = ggml_new_graph_custom(ggml, 128, false);
    ggml_build_forward_expand(graph, output.tensor);
    auto buffer = ggml_backend_alloc_ctx_tensors(ggml, backend);
    if (!buffer) throw std::runtime_error("tensor allocation failed");
    std::vector<float> values(64);
    for (size_t i = 0; i < values.size(); ++i) values[i] = std::sin(float(i) * 0.13f);
    const int32_t indices[] = {3, 11};
    ggml_backend_tensor_set(input.tensor, values.data(), 0, values.size() * sizeof(float));
    ggml_backend_tensor_set(positions.tensor, sliced ? indices : indices + 1, 0, (sliced ? 2 : 1) * sizeof(int32_t));
    if (core::compute_backend_graph(backend, graph) != GGML_STATUS_SUCCESS) throw std::runtime_error("compute failed");
    ggml_backend_synchronize(backend);
    ggml_backend_tensor_get(output.tensor, values.data(), 0, values.size() * sizeof(float));
    core::release_backend_graph_resources(backend, graph);
    ggml_backend_buffer_free(buffer);
    ggml_free(ggml);
    ggml_backend_free(backend);
    return values;
}

int main(int argc, char ** argv) {
    try {
        const auto type = argc > 1 && std::string(argv[1]) == "--vulkan"
            ? engine::core::BackendType::Vulkan : engine::core::BackendType::Cpu;
        const auto expected = run(engine::core::BackendType::Cpu, false);
        const auto actual = run(type, true);
        for (size_t i = 0; i < actual.size(); ++i) {
            if (!std::isfinite(actual[i]) || std::fabs(actual[i] - expected[i]) > 0.0001f) {
                throw std::runtime_error("RoPE view changed position or output at index " + std::to_string(i));
            }
        }
        std::cout << "PASS: sliced RoPE position matches independent CPU position\n";
        return 0;
    } catch (const std::exception & error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
