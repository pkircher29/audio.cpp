#include "engine/framework/core/backend.h"
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

// Output crosses Vulkan's guaranteed 65535 workgroups * 256 threads.
// Overlapping kernels and two channels exercise both indexing dimensions.
std::vector<float> run(engine::core::BackendType type) {
    using namespace engine;
    auto backend = core::init_backend({type, 0, 2});
    if (!backend) throw std::runtime_error("backend unavailable");
    auto * ctx = ggml_init({4 * 1024 * 1024, nullptr, true});
    auto * input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 8, 4200001);
    auto * output = ggml_col2im_1d(ctx, input, 2, 2, 0);
    auto * graph = ggml_new_graph_custom(ctx, 128, false);
    ggml_build_forward_expand(graph, output);
    auto buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buffer) throw std::runtime_error("tensor allocation failed");
    std::vector<float> values(ggml_nelements(input));
    for (size_t i = 0; i < values.size(); ++i) values[i] = float(int(i % 31) - 15) / 16.0f;
    ggml_backend_tensor_set(input, values.data(), 0, ggml_nbytes(input));
    if (core::compute_backend_graph(backend, graph) != GGML_STATUS_SUCCESS) throw std::runtime_error("compute failed");
    ggml_backend_synchronize(backend);
    values.resize(ggml_nelements(output));
    ggml_backend_tensor_get(output, values.data(), 0, ggml_nbytes(output));
    core::release_backend_graph_resources(backend, graph);
    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    ggml_backend_free(backend);
    return values;
}

int main(int argc, char ** argv) {
    try {
        if (argc != 2 || std::string(argv[1]) != "--vulkan")
            throw std::runtime_error("Use --vulkan; this regression requires a GPU backend");
        const auto type = engine::core::BackendType::Vulkan;
        // Independent scatter-add reference; CPU backend does not implement this op.
        std::vector<float> expected(8400004 * 2, 0.0f);
        for (size_t t = 0; t < 4200001; ++t)
            for (size_t channel = 0; channel < 2; ++channel)
                for (size_t k = 0; k < 4; ++k) {
                    const size_t index = t * 8 + channel * 4 + k;
                    expected[channel * 8400004 + t * 2 + k] += float(int(index % 31) - 15) / 16.0f;
                }
        const auto actual = run(type);
        for (size_t i = 0; i < actual.size(); ++i) {
            if (!std::isfinite(actual[i]) || std::fabs(actual[i] - expected[i]) > 0.0001f)
                throw std::runtime_error("col2im mismatch at " + std::to_string(i));
        }
        std::cout << "PASS: large col2im dispatch matches scatter-add reference across " << actual.size() << " outputs\n";
        return 0;
    } catch (const std::exception & error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
