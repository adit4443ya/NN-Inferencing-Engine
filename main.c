#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <jansson.h>

/**
 * Neural Network Inference Engine
 * 
 * A high-performance, fixed-point arithmetic neural network inference engine
 * designed for embedded systems and IoT devices with future parallelization in mind.
 * 
 * Features:
 * - Configurable fixed-point arithmetic for embedded systems without FPU
 * - Dynamic network configuration from JSON
 * - Layer-independent computation (parallelization-ready)
 * - Efficient memory management with memory pools
 * - Proper activation function implementation
 * - Batch processing support
 */

// Configuration
#define MAX_LAYERS 16                  // Maximum number of layers supported
#define MAX_NEURONS_PER_LAYER 4096     // Maximum neurons per layer
#define MAX_BATCH_SIZE 128             // Maximum batch size for inference
int NUM_INPUT_FEATURES = 4096;        // Maximum input features supported
#define MAX_SAMPLES 10000              // Maximum number of test samples
int num_samples = 2500;

// Memory alignment for better cache utilization
#define ALIGNMENT 64

// Fixed-point arithmetic configuration
// Q12.20 format gives good balance between range (-2048 to 2048) and precision (0.000001)
#define FIXED_POINT_ENABLE 1           // Set to 0 for float arithmetic (debugging)
#define FIXED_POINT_SCALE (1LL << 20)  // 20-bit fractional part

// Fixed-point arithmetic types and macros
#if FIXED_POINT_ENABLE
    typedef int32_t fixed_t;               // 32-bit fixed-point value
    typedef int64_t fixed_acc_t;           // 64-bit accumulator for calculations
    
    // Conversion macros
    #define FLOAT_TO_FIXED(x) ((fixed_t)((x) * FIXED_POINT_SCALE))
    #define FIXED_TO_FLOAT(x) ((float)(x) / FIXED_POINT_SCALE)
    
    // Arithmetic macros
    #define FIXED_MUL(x, y) ((fixed_t)(((fixed_acc_t)(x) * (fixed_acc_t)(y)) / FIXED_POINT_SCALE))
    #define FIXED_DIV(x, y) ((fixed_t)(((fixed_acc_t)(x) * FIXED_POINT_SCALE) / (fixed_acc_t)(y)))
    
    // Clamping to avoid overflow
    #define FIXED_CLAMP(x, min, max) ((x) < (min) ? (min) : ((x) > (max) ? (max) : (x)))
    #define FIXED_SATURATE(x) FIXED_CLAMP(x, INT32_MIN, INT32_MAX)
#else
    typedef float fixed_t;                 // Use floats for debugging
    typedef double fixed_acc_t;            // Higher precision for accumulators
    
    // Conversion macros (no-op for float mode)
    #define FLOAT_TO_FIXED(x) (x)
    #define FIXED_TO_FLOAT(x) (x)
    
    // Arithmetic macros
    #define FIXED_MUL(x, y) ((x) * (y))
    #define FIXED_DIV(x, y) ((x) / (y))
    
    // Clamping
    #define FIXED_CLAMP(x, min, max) ((x) < (min) ? (min) : ((x) > (max) ? (max) : (x)))
    #define FIXED_SATURATE(x) FIXED_CLAMP(x, -FLT_MAX, FLT_MAX)
#endif

// Logging macros
#define LOG(fmt, ...) fprintf(stderr, "[INFO] %s:%d: " fmt "\n", __FILE__, __LINE__, ##__VA_ARGS__)
#define WARN(fmt, ...) fprintf(stderr, "[WARN] %s:%d: " fmt "\n", __FILE__, __LINE__, ##__VA_ARGS__)
#define ERR(fmt, ...) fprintf(stderr, "[ERROR] %s:%d: " fmt "\n", __FILE__, __LINE__, ##__VA_ARGS__)
#define DEBUG(fmt, ...) fprintf(stderr, "[DEBUG] %s:%d: " fmt "\n", __FILE__, __LINE__, ##__VA_ARGS__)

// Memory pool for efficient allocation
typedef struct {
    void *buffer;
    size_t size;
    size_t used;
} MemoryPool;

// Activation function types
typedef enum {
    ACTIVATION_NONE,
    ACTIVATION_RELU,
    ACTIVATION_SIGMOID,
    ACTIVATION_TANH,
    ACTIVATION_SOFTMAX,
    ACTIVATION_LEAKY_RELU
} ActivationType;

// Layer types
typedef enum {
    LAYER_DENSE,
    LAYER_CONV2D,        // Placeholder for future implementation
    LAYER_MAXPOOL,       // Placeholder for future implementation
    LAYER_FLATTEN        // Placeholder for future implementation
} LayerType;

// Forward declarations for layer-specific structures
typedef struct DenseLayer DenseLayer;

// Generic layer interface
typedef struct {
    LayerType type;
    int input_size;
    int output_size;
    ActivationType activation;
    void *layer_data;       // Type-specific data (e.g., DenseLayer)
    
    // Function pointers for layer operations (for polymorphic behavior)
    void (*forward)(struct Layer *layer, fixed_t *input, fixed_t *output, int batch_size);
    void (*cleanup)(struct Layer *layer);
} Layer;

// Dense (fully connected) layer specific data
struct DenseLayer {
    fixed_t *weights;    // [output_size][input_size]
    fixed_t *biases;     // [output_size]
};

// Neural network structure
typedef struct {
    int num_layers;
    Layer **layers;
    int input_size;
    int output_size;
    fixed_t **activations;  // Intermediate activations between layers
    MemoryPool *pool;       // Memory pool for allocations
} Network;

// Function prototypes
MemoryPool* create_memory_pool(size_t size);
void* pool_alloc(MemoryPool *pool, size_t size);
void* pool_aligned_alloc(MemoryPool *pool, size_t size, size_t alignment);
void free_memory_pool(MemoryPool *pool);
Network* create_network(const char *config_file, const char *weights_file);
void free_network(Network *network);
void forward_pass(Network *network, fixed_t *input, fixed_t *output, int batch_size);
int get_prediction(fixed_t *output, int size);
void load_samples(fixed_t *inputs, int *labels, int num_samples, const char *inputs_file, const char *labels_file);
void apply_relu(fixed_t *data, int size);
void apply_softmax(fixed_t *data, int size);
void dense_layer_forward(Layer *layer, fixed_t *input, fixed_t *output, int batch_size);
void dense_layer_cleanup(Layer *layer);

// Fixed-point activation functions
fixed_t fixed_relu(fixed_t x) {
    return x > 0 ? x : 0;
}

fixed_t fixed_sigmoid(fixed_t x) {
    // Protect against overflows
    if (x < FLOAT_TO_FIXED(-10.0f)) return 0;
    if (x > FLOAT_TO_FIXED(10.0f)) return FIXED_POINT_SCALE;
    
    // Use the sigmoid approximation: sigmoid(x) = 1 / (1 + exp(-x))
    float fx = FIXED_TO_FLOAT(x);
    float sigmoid = 1.0f / (1.0f + expf(-fx));
    return FLOAT_TO_FIXED(sigmoid);
}

fixed_t fixed_tanh(fixed_t x) {
    // Protect against overflows
    if (x < FLOAT_TO_FIXED(-10.0f)) return FLOAT_TO_FIXED(-1.0f);
    if (x > FLOAT_TO_FIXED(10.0f)) return FLOAT_TO_FIXED(1.0f);
    
    // Use tanh approximation: tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))
    float fx = FIXED_TO_FLOAT(x);
    float tanh_val = tanhf(fx);
    return FLOAT_TO_FIXED(tanh_val);
}

fixed_t fixed_leaky_relu(fixed_t x, fixed_t alpha) {
    return x > 0 ? x : FIXED_MUL(alpha, x);
}

// Apply ReLU activation to an array of values
void apply_relu(fixed_t *data, int size) {
    for (int i = 0; i < size; i++) {
        data[i] = fixed_relu(data[i]);
    }
}

// Apply softmax activation to an array of values
void apply_softmax(fixed_t *data, int size) {
    // Find maximum value for numerical stability
    fixed_t max_val = data[0];
    for (int i = 1; i < size; i++) {
        if (data[i] > max_val) {
            max_val = data[i];
        }
    }
    
    // Compute exp(x_i - max) for each element
    fixed_acc_t sum = 0;
    for (int i = 0; i < size; i++) {
        // Compute exp(x - max) using float for accuracy
        float val = FIXED_TO_FLOAT(data[i] - max_val);
        
        // Clamp to avoid overflow
        if (val > 88.0f) val = 88.0f;  // exp(88) is close to float max
        if (val < -88.0f) val = -88.0f;
        
        // Convert back to fixed point
        data[i] = FLOAT_TO_FIXED(expf(val));
        sum += data[i];
    }
    
    // Normalize by the sum
    if (sum == 0) sum = 1;  // Avoid division by zero
    for (int i = 0; i < size; i++) {
        data[i] = FIXED_DIV(data[i], sum);
    }
}

// Create a memory pool with the specified size
MemoryPool* create_memory_pool(size_t size) {
    MemoryPool *pool = (MemoryPool *)malloc(sizeof(MemoryPool));
    if (!pool) {
        ERR("Failed to allocate memory pool structure");
        return NULL;
    }
    
    pool->buffer = malloc(size);
    if (!pool->buffer) {
        ERR("Failed to allocate memory pool buffer of size %zu", size);
        free(pool);
        return NULL;
    }
    
    pool->size = size;
    pool->used = 0;
    
    return pool;
}

// Allocate memory from the pool
void* pool_alloc(MemoryPool *pool, size_t size) {
    if (pool->used + size > pool->size) {
        ERR("Memory pool exhausted: requested %zu bytes, available %zu bytes", 
            size, pool->size - pool->used);
        return NULL;
    }
    
    void *ptr = (uint8_t *)pool->buffer + pool->used;
    pool->used += size;
    
    // Ensure 8-byte alignment for next allocation
    pool->used = (pool->used + 7) & ~7;
    
    return ptr;
}

// Allocate aligned memory from the pool
void* pool_aligned_alloc(MemoryPool *pool, size_t size, size_t alignment) {
    // Calculate padding needed for alignment
    size_t padding = (alignment - (pool->used % alignment)) % alignment;
    size_t total_size = size + padding;
    
    if (pool->used + total_size > pool->size) {
        ERR("Memory pool exhausted: requested %zu bytes (aligned), available %zu bytes", 
            total_size, pool->size - pool->used);
        return NULL;
    }
    
    pool->used += padding;  // Skip bytes for alignment
    void *ptr = (uint8_t *)pool->buffer + pool->used;
    pool->used += size;
    
    return ptr;
}

// Free the memory pool
void free_memory_pool(MemoryPool *pool) {
    if (pool) {
        if (pool->buffer) {
            free(pool->buffer);
        }
        free(pool);
    }
}

// Dense layer forward pass
void dense_layer_forward(Layer *layer, fixed_t *input, fixed_t *output, int batch_size) {
    DenseLayer *dense = (DenseLayer *)layer->layer_data;
    int input_size = layer->input_size;
    int output_size = layer->output_size;
    
    // Process all samples in the batch
    for (int batch = 0; batch < batch_size; batch++) {
        fixed_t *batch_input = input + batch * input_size;
        fixed_t *batch_output = output + batch * output_size;
        
        // For each output neuron
        for (int out = 0; out < output_size; out++) {
            fixed_acc_t sum = 0;
            
            // For each input neuron
            for (int in = 0; in < input_size; in++) {
                // Get weight at [out][in]
                fixed_t weight = dense->weights[out * input_size + in];
                
                // Accumulate weighted sum
                sum += (fixed_acc_t)weight * batch_input[in];
            }
            
            // Apply bias and normalize (divide by FIXED_POINT_SCALE to restore fixed-point format)
            sum = sum / FIXED_POINT_SCALE + dense->biases[out];
            
            // Clamp result to avoid overflow
            batch_output[out] = FIXED_SATURATE(sum);
        }
        
        // Apply activation function
        switch (layer->activation) {
            case ACTIVATION_RELU:
                apply_relu(batch_output, output_size);
                break;
                
            case ACTIVATION_SOFTMAX:
                apply_softmax(batch_output, output_size);
                break;
                
            case ACTIVATION_NONE:
            default:
                // No activation function
                break;
        }
    }
}

// Dense layer cleanup
void dense_layer_cleanup(Layer *layer) {
    // No cleanup needed for DenseLayer as its resources are pool-allocated
    // This would be used if we had dynamically allocated resources
}

// Load network configuration from JSON file
Network* create_network(const char *config_file, const char *weights_file) {
    // Allocate initial memory pool (we'll reallocate with correct size once we know the network structure)
    size_t initial_pool_size = 1024 * 1024;  // 1MB initial size
    MemoryPool *temp_pool = create_memory_pool(initial_pool_size);
    if (!temp_pool) {
        ERR("Failed to create temporary memory pool");
        return NULL;
    }
    
    // Parse configuration file
    json_error_t error;
    json_t *config = json_load_file(config_file, 0, &error);
    if (!config) {
        ERR("Failed to parse configuration file: %s", error.text);
        free_memory_pool(temp_pool);
        return NULL;
    }
    
    // Extract basic network information
    json_t *jlayer_sizes = json_object_get(config, "layer_sizes");
    json_t *jactivations = json_object_get(config, "activations");
    json_t *jnum_layers = json_object_get(config, "num_layers");
    
    if (!json_is_array(jlayer_sizes) || !json_is_array(jactivations) || !json_is_integer(jnum_layers)) {
        ERR("Invalid configuration format. Expected 'layer_sizes' and 'activations' arrays, and 'num_layers' integer.");
        json_decref(config);
        free_memory_pool(temp_pool);
        return NULL;
    }
    
    int num_layers = json_integer_value(jnum_layers) - 1;  // Subtract 1 because layer_sizes includes input layer
    
    if (num_layers <= 0 || num_layers > MAX_LAYERS) {
        ERR("Invalid number of layers: %d. Must be between 1 and %d.", num_layers, MAX_LAYERS);
        json_decref(config);
        free_memory_pool(temp_pool);
        return NULL;
    }
    
    // Allocate initial network structure (temporary)
    Network *network = (Network *)malloc(sizeof(Network));
    if (!network) {
        ERR("Failed to allocate network structure");
        json_decref(config);
        free_memory_pool(temp_pool);
        return NULL;
    }
    
    network->num_layers = num_layers;
    network->layers = (Layer **)malloc(num_layers * sizeof(Layer *));
    if (!network->layers) {
        ERR("Failed to allocate layers array");
        free(network);
        json_decref(config);
        free_memory_pool(temp_pool);
        return NULL;
    }
    
    // Extract layer sizes
    int *layer_sizes = (int *)malloc((num_layers + 1) * sizeof(int));
    if (!layer_sizes) {
        ERR("Failed to allocate layer sizes array");
        free(network->layers);
        free(network);
        json_decref(config);
        free_memory_pool(temp_pool);
        return NULL;
    }
    
    for (int i = 0; i <= num_layers; i++) {
        json_t *jsize = json_array_get(jlayer_sizes, i);
        if (!json_is_integer(jsize)) {
            ERR("Invalid layer size at index %d", i);
            free(layer_sizes);
            free(network->layers);
            free(network);
            json_decref(config);
            free_memory_pool(temp_pool);
            return NULL;
        }
        
        int size = json_integer_value(jsize);
        if (size <= 0 || size > MAX_NEURONS_PER_LAYER) {
            ERR("Invalid layer size: %d. Must be between 1 and %d.", size, MAX_NEURONS_PER_LAYER);
            free(layer_sizes);
            free(network->layers);
            free(network);
            json_decref(config);
            free_memory_pool(temp_pool);
            return NULL;
        }
        
        layer_sizes[i] = size;
    }
    NUM_INPUT_FEATURES=layer_sizes[0];
    
    network->input_size = layer_sizes[0];
    network->output_size = layer_sizes[num_layers];
    
    // Calculate total memory requirements
    size_t weights_size = 0;
    size_t activations_size = 0;
    
    for (int i = 0; i < num_layers; i++) {
        int input_size = layer_sizes[i];
        int output_size = layer_sizes[i + 1];
        
        // Size for weights and biases
        weights_size += (input_size * output_size + output_size) * sizeof(fixed_t);
        
        // Size for activations (including input and output)
        if (i == 0) {
            activations_size += input_size * MAX_BATCH_SIZE * sizeof(fixed_t);
        }
        activations_size += output_size * MAX_BATCH_SIZE * sizeof(fixed_t);
    }
    
    // Add size for network structure and layers
    size_t struct_size = sizeof(Network) + num_layers * sizeof(Layer) + 
                      (num_layers + 1) * sizeof(fixed_t *);
    
    // Calculate total memory pool size with padding
    size_t total_pool_size = struct_size + weights_size + activations_size;
    total_pool_size = (total_pool_size * 12) / 10;  // Add 20% extra space
    
    // Create final memory pool
    MemoryPool *pool = create_memory_pool(total_pool_size);
    if (!pool) {
        ERR("Failed to create memory pool of size %zu", total_pool_size);
        free(layer_sizes);
        free(network->layers);
        free(network);
        json_decref(config);
        free_memory_pool(temp_pool);
        return NULL;
    }
    
    // Allocate final network structure from pool
    Network *final_network = (Network *)pool_alloc(pool, sizeof(Network));
    if (!final_network) {
        ERR("Failed to allocate network structure from pool");
        free(layer_sizes);
        free(network->layers);
        free(network);
        json_decref(config);
        free_memory_pool(temp_pool);
        free_memory_pool(pool);
        return NULL;
    }
    
    final_network->num_layers = num_layers;
    final_network->input_size = layer_sizes[0];
    final_network->output_size = layer_sizes[num_layers];
    final_network->pool = pool;
    
    // Allocate layers array from pool
    final_network->layers = (Layer **)pool_alloc(pool, num_layers * sizeof(Layer *));
    if (!final_network->layers) {
        ERR("Failed to allocate layers array from pool");
        free(layer_sizes);
        free(network->layers);
        free(network);
        json_decref(config);
        free_memory_pool(temp_pool);
        free_memory_pool(pool);
        return NULL;
    }
    
    // Allocate activations array from pool
    final_network->activations = (fixed_t **)pool_alloc(pool, (num_layers + 1) * sizeof(fixed_t *));
    if (!final_network->activations) {
        ERR("Failed to allocate activations array from pool");
        free(layer_sizes);
        free(network->layers);
        free(network);
        json_decref(config);
        free_memory_pool(temp_pool);
        free_memory_pool(pool);
        return NULL;
    }
    
    // Allocate space for each activation
    for (int i = 0; i <= num_layers; i++) {
        int size = layer_sizes[i];
        final_network->activations[i] = (fixed_t *)pool_aligned_alloc(pool, 
                                        size * MAX_BATCH_SIZE * sizeof(fixed_t), 
                                        ALIGNMENT);
        if (!final_network->activations[i]) {
            ERR("Failed to allocate activations for layer %d", i);
            free(layer_sizes);
            free(network->layers);
            free(network);
            json_decref(config);
            free_memory_pool(temp_pool);
            free_memory_pool(pool);
            return NULL;
        }
    }
    
    // Open weights file
    FILE *weights_file_handle = fopen(weights_file, "rb");
    if (!weights_file_handle) {
        ERR("Failed to open weights file: %s", weights_file);
        free(layer_sizes);
        free(network->layers);
        free(network);
        json_decref(config);
        free_memory_pool(temp_pool);
        free_memory_pool(pool);
        return NULL;
    }
    
    // Create each layer
    for (int i = 0; i < num_layers; i++) {
        // Allocate layer from pool
        Layer *layer = (Layer *)pool_alloc(pool, sizeof(Layer));
        if (!layer) {
            ERR("Failed to allocate layer %d", i);
            fclose(weights_file_handle);
            free(layer_sizes);
            free(network->layers);
            free(network);
            json_decref(config);
            free_memory_pool(temp_pool);
            free_memory_pool(pool);
            return NULL;
        }
        
        final_network->layers[i] = layer;
        
        // Set layer properties
        layer->type = LAYER_DENSE;  // Currently only supporting dense layers
        layer->input_size = layer_sizes[i];
        layer->output_size = layer_sizes[i + 1];
        
        // Determine activation function
        json_t *jactivation = json_array_get(jactivations, i);
        if (!json_is_string(jactivation)) {
            ERR("Invalid activation type at index %d", i);
            fclose(weights_file_handle);
            free(layer_sizes);
            free(network->layers);
            free(network);
            json_decref(config);
            free_memory_pool(temp_pool);
            free_memory_pool(pool);
            return NULL;
        }
        
        const char *activation_str = json_string_value(jactivation);
        if (strcmp(activation_str, "relu") == 0) {
            layer->activation = ACTIVATION_RELU;
        } else if (strcmp(activation_str, "sigmoid") == 0) {
            layer->activation = ACTIVATION_SIGMOID;
        } else if (strcmp(activation_str, "tanh") == 0) {
            layer->activation = ACTIVATION_TANH;
        } else if (strcmp(activation_str, "softmax") == 0) {
            layer->activation = ACTIVATION_SOFTMAX;
        } else if (strcmp(activation_str, "leaky_relu") == 0) {
            layer->activation = ACTIVATION_LEAKY_RELU;
        } else {
            layer->activation = ACTIVATION_NONE;
        }
        
        // Set function pointers
        layer->forward = dense_layer_forward;
        layer->cleanup = dense_layer_cleanup;
        
        // Allocate and initialize dense layer data
        DenseLayer *dense = (DenseLayer *)pool_alloc(pool, sizeof(DenseLayer));
        if (!dense) {
            ERR("Failed to allocate dense layer data for layer %d", i);
            fclose(weights_file_handle);
            free(layer_sizes);
            free(network->layers);
            free(network);
            json_decref(config);
            free_memory_pool(temp_pool);
            free_memory_pool(pool);
            return NULL;
        }
        
        layer->layer_data = dense;
        
        // Allocate weights and biases
        int weights_count = layer->input_size * layer->output_size;
        dense->weights = (fixed_t *)pool_aligned_alloc(pool, weights_count * sizeof(fixed_t), ALIGNMENT);
        dense->biases = (fixed_t *)pool_aligned_alloc(pool, layer->output_size * sizeof(fixed_t), ALIGNMENT);
        
        if (!dense->weights || !dense->biases) {
            ERR("Failed to allocate weights or biases for layer %d", i);
            fclose(weights_file_handle);
            free(layer_sizes);
            free(network->layers);
            free(network);
            json_decref(config);
            free_memory_pool(temp_pool);
            free_memory_pool(pool);
            return NULL;
        }
        
        // Read weights from file
        // Note: The weights are stored in row-major order (output_size x input_size)
        float *temp_weights = (float *)malloc(weights_count * sizeof(float));
        if (!temp_weights) {
            ERR("Failed to allocate temporary weights buffer");
            fclose(weights_file_handle);
            free(layer_sizes);
            free(network->layers);
            free(network);
            json_decref(config);
            free_memory_pool(temp_pool);
            free_memory_pool(pool);
            return NULL;
        }
        
        size_t read_count = fread(temp_weights, sizeof(float), weights_count, weights_file_handle);
        if (read_count != weights_count) {
            ERR("Failed to read weights for layer %d. Expected %d, got %zu", 
                i, weights_count, read_count);
            free(temp_weights);
            fclose(weights_file_handle);
            free(layer_sizes);
            free(network->layers);
            free(network);
            json_decref(config);
            free_memory_pool(temp_pool);
            free_memory_pool(pool);
            return NULL;
        }
        
        // Convert weights to fixed-point
        for (int j = 0; j < weights_count; j++) {
            dense->weights[j] = FLOAT_TO_FIXED(temp_weights[j]);
        }
        free(temp_weights);
        
        // Read biases from file
        float *temp_biases = (float *)malloc(layer->output_size * sizeof(float));
        if (!temp_biases) {
            ERR("Failed to allocate temporary biases buffer");
            fclose(weights_file_handle);
            free(layer_sizes);
            free(network->layers);
            free(network);
            json_decref(config);
            free_memory_pool(temp_pool);
            free_memory_pool(pool);
            return NULL;
        }
        
        read_count = fread(temp_biases, sizeof(float), layer->output_size, weights_file_handle);
        if (read_count != layer->output_size) {
            ERR("Failed to read biases for layer %d. Expected %d, got %zu", 
                i, layer->output_size, read_count);
            free(temp_biases);
            fclose(weights_file_handle);
            free(layer_sizes);
            free(network->layers);
            free(network);
            json_decref(config);
            free_memory_pool(temp_pool);
            free_memory_pool(pool);
            return NULL;
        }
        
        // Convert biases to fixed-point
        for (int j = 0; j < layer->output_size; j++) {
            dense->biases[j] = FLOAT_TO_FIXED(temp_biases[j]);
        }
        free(temp_biases);
    }
    
    // Cleanup
    fclose(weights_file_handle);
    free(layer_sizes);
    free(network->layers);
    free(network);
    json_decref(config);
    free_memory_pool(temp_pool);
    
    LOG("Network loaded successfully: %d layers, input size %d, output size %d", 
        final_network->num_layers, final_network->input_size, final_network->output_size);
    
    return final_network;
}

// Free network resources
void free_network(Network *network) {
    if (!network) return;
    
    // Free memory pool (which contains all network allocations)
    free_memory_pool(network->pool);
}
// Run forward pass through the network
void forward_pass(Network *network, fixed_t *input, fixed_t *output, int batch_size) {
    if (!network || !input || !output || batch_size <= 0) {
        ERR("Invalid parameters for forward pass");
        return;
    }
    
    if (batch_size > MAX_BATCH_SIZE) {
        WARN("Batch size %d exceeds maximum %d, truncating", batch_size, MAX_BATCH_SIZE);
        batch_size = MAX_BATCH_SIZE;
    }
    
    // Copy input to first activation buffer
    memcpy(network->activations[0], input, network->input_size * batch_size * sizeof(fixed_t));
    
    // Process each layer sequentially
    for (int i = 0; i < network->num_layers; i++) {
        Layer *layer = network->layers[i];
        layer->forward(layer, network->activations[i], network->activations[i + 1], batch_size);
    }
    
    // Copy final activations to output
    memcpy(output, network->activations[network->num_layers], 
           network->output_size * batch_size * sizeof(fixed_t));
}

// Get predicted class index from output probabilities
int get_prediction(fixed_t *output, int size) {
    if (!output || size <= 0) {
        ERR("Invalid parameters for get_prediction");
        return -1;
    }
    
    int max_idx = 0;
    fixed_t max_val = output[0];
    
    for (int i = 1; i < size; i++) {
        if (output[i] > max_val) {
            max_val = output[i];
            max_idx = i;
        }
    }
    
    return max_idx;
}

// Load sample inputs and labels from binary files
void load_samples(fixed_t *inputs, int *labels, int num_samples, const char *inputs_file, const char *labels_file) {
    if (!inputs || !labels || num_samples <= 0) {
        ERR("Invalid parameters for load_samples");
        return;
    }
    
    // Open input samples file
    FILE *f_inputs = fopen(inputs_file, "rb");
    if (!f_inputs) {
        ERR("Failed to open inputs file: %s", inputs_file);
        return;
    }
    
    // Open labels file
    FILE *f_labels = fopen(labels_file, "rb");
    if (!f_labels) {
        ERR("Failed to open labels file: %s", labels_file);
        fclose(f_inputs);
        return;
    }
    
    // Allocate temporary buffer for float values
    float *temp_inputs = (float *)malloc(num_samples * NUM_INPUT_FEATURES * sizeof(float));
    if (!temp_inputs) {
        ERR("Failed to allocate temporary buffer for inputs");
        fclose(f_inputs);
        fclose(f_labels);
        return;
    }
    
    // Read inputs as float values
    size_t inputs_read = fread(temp_inputs, sizeof(float), num_samples * NUM_INPUT_FEATURES, f_inputs);
    if (inputs_read != num_samples * NUM_INPUT_FEATURES && !feof(f_inputs)) {
        ERR("Failed to read all inputs: expected %d, got %zu", 
            num_samples * NUM_INPUT_FEATURES, inputs_read);
        free(temp_inputs);
        fclose(f_inputs);
        fclose(f_labels);
        return;
    }
    
    // Convert float inputs to fixed-point
    for (int i = 0; i < inputs_read; i++) {
        inputs[i] = FLOAT_TO_FIXED(temp_inputs[i]);
    }
    free(temp_inputs);
    
    // Read integer labels directly
    size_t labels_read = fread(labels, sizeof(int), num_samples, f_labels);
    if (labels_read != num_samples && !feof(f_labels)) {
        ERR("Failed to read all labels: expected %d, got %zu", 
            num_samples, labels_read);
        fclose(f_inputs);
        fclose(f_labels);
        return;
    }
    
    fclose(f_inputs);
    fclose(f_labels);
    
    LOG("Loaded %zu inputs and %zu labels successfully", inputs_read / NUM_INPUT_FEATURES, labels_read);
}

// Print model summary
void print_model_summary(Network *network) {
    printf("=== Neural Network Summary ===\n");
    printf("Number of layers: %d\n", network->num_layers);
    printf("Input size: %d\n", network->input_size);
    printf("Output size: %d\n", network->output_size);
    
    printf("\nLayer details:\n");
    for (int i = 0; i < network->num_layers; i++) {
        Layer *layer = network->layers[i];
        printf("Layer %d: ", i + 1);
        
        switch (layer->type) {
            case LAYER_DENSE:
                printf("Dense (%d -> %d) ", layer->input_size, layer->output_size);
                break;
            case LAYER_CONV2D:
                printf("Conv2D ");
                break;
            case LAYER_MAXPOOL:
                printf("MaxPool ");
                break;
            case LAYER_FLATTEN:
                printf("Flatten ");
                break;
            default:
                printf("Unknown ");
        }
        
        printf("with ");
        switch (layer->activation) {
            case ACTIVATION_NONE:
                printf("no activation");
                break;
            case ACTIVATION_RELU:
                printf("ReLU activation");
                break;
            case ACTIVATION_SIGMOID:
                printf("Sigmoid activation");
                break;
            case ACTIVATION_TANH:
                printf("Tanh activation");
                break;
            case ACTIVATION_SOFTMAX:
                printf("Softmax activation");
                break;
            case ACTIVATION_LEAKY_RELU:
                printf("Leaky ReLU activation");
                break;
            default:
                printf("unknown activation");
        }
        
        printf("\n");
    }
    printf("=============================\n");
}

// Run inference on a dataset and print evaluation metrics
void evaluate_model(Network *network, fixed_t *inputs, int *labels, int num_samples) {    
    // Allocate output buffer
    fixed_t *outputs = (fixed_t *)malloc(network->output_size * sizeof(fixed_t));
    if (!outputs) {
        ERR("Failed to allocate output buffer");
        return;
    }
    
    int correct = 0;
    int batch_size = 1;  // Process one sample at a time for simplicity
    
    LOG("Running inference on %d samples...", num_samples);
    
    for (int i = 0; i < num_samples; i++) {
        // Get input for this sample
        fixed_t *sample_input = inputs + i * network->input_size;
        
        // Run forward pass
        forward_pass(network, sample_input, outputs, batch_size);
        
        // Get prediction
        int prediction = get_prediction(outputs, network->output_size);
        
        // Compare with true label
        if (prediction == labels[i]) {
            correct++;
        }
        
        // Print progress every 100 samples
        if ((i + 1) % 100 == 0 || i == num_samples - 1) {
            LOG("Processed %d/%d samples, accuracy so far: %.2f%%", 
                i + 1, num_samples, (float)correct / (i + 1) * 100.0f);
        }
    }
    
    // Calculate final accuracy
    float accuracy = (float)correct / num_samples * 100.0f;
    printf("\n===== Evaluation Results =====\n");
    printf("Total samples: %d\n", num_samples);
    printf("Correct predictions: %d\n", correct);
    printf("Accuracy: %.2f%%\n", accuracy);
    printf("=============================\n");
    
    free(outputs);
}

// Run inference on batches of data
void batch_inference(Network *network, fixed_t *inputs, int *labels, int num_samples, int batch_size) {
    if (batch_size > MAX_BATCH_SIZE) {
        WARN("Batch size %d exceeds maximum %d, truncating", batch_size, MAX_BATCH_SIZE);
        batch_size = MAX_BATCH_SIZE;
    }
    
    // Allocate output buffer for entire batch
    fixed_t *batch_outputs = (fixed_t *)malloc(network->output_size * batch_size * sizeof(fixed_t));
    if (!batch_outputs) {
        ERR("Failed to allocate batch output buffer");
        return;
    }
    
    int correct = 0;
    int sample_idx = 0;
    
    LOG("Running batch inference with batch size %d on %d samples...", batch_size, num_samples);
    
    while (sample_idx < num_samples) {
        // Determine actual batch size for this iteration
        int current_batch_size = batch_size;
        if (sample_idx + batch_size > num_samples) {
            current_batch_size = num_samples - sample_idx;
        }
        
        // Get input batch
        fixed_t *batch_input = inputs + sample_idx * network->input_size;
        
        // Run forward pass for the batch
        forward_pass(network, batch_input, batch_outputs, current_batch_size);
        
        // Process predictions for the batch
        for (int i = 0; i < current_batch_size; i++) {
            fixed_t *sample_output = batch_outputs + i * network->output_size;
            int prediction = get_prediction(sample_output, network->output_size);
            
            if (prediction == labels[sample_idx + i]) {
                correct++;
            }
        }
        
        // Update sample index
        sample_idx += current_batch_size;
        
        // Print progress
        if (sample_idx % (10 * batch_size) == 0 || sample_idx >= num_samples) {
            LOG("Processed %d/%d samples, accuracy so far: %.2f%%", 
                sample_idx, num_samples, (float)correct / sample_idx * 100.0f);
        }
    }
    
    // Calculate final accuracy
    float accuracy = (float)correct / num_samples * 100.0f;
    printf("\n===== Batch Inference Results =====\n");
    printf("Total samples: %d\n", num_samples);
    printf("Batch size: %d\n", batch_size);
    printf("Correct predictions: %d\n", correct);
    printf("Accuracy: %.2f%%\n", accuracy);
    printf("===================================\n");
    
    free(batch_outputs);
}

int main(int argc, char *argv[]) {
    printf("Neural Network Inference Engine\n");
    printf("===============================\n");
    
    // Default file paths
    const char *config_file = "model_config.json";
    const char *weights_file = "weights.bin";
    const char *inputs_file = "sample_inputs.bin";
    const char *labels_file = "sample_labels.bin";
      // Default number of samples to process
    int batch_size = 32;     // Default batch size
    
    // Override defaults from command line if provided
    if (argc > 1) config_file = argv[1];
    if (argc > 2) weights_file = argv[2];
    if (argc > 3) inputs_file = argv[3];
    if (argc > 4) labels_file = argv[4];
    if (argc > 5) num_samples = atoi(argv[5]);
    if (argc > 6) batch_size = atoi(argv[6]);
    
    // Validate parameters
    if (num_samples <= 0 || num_samples > MAX_SAMPLES) {
        ERR("Invalid number of samples: %d. Using default: %d", num_samples, 2000);
        num_samples = 2000;
    }
    
    if (batch_size <= 0 || batch_size > MAX_BATCH_SIZE) {
        ERR("Invalid batch size: %d. Using default: %d", batch_size, 32);
        batch_size = 32;
    }
    
    // Log configuration
    LOG("Configuration:");
    LOG("  Model config: %s", config_file);
    LOG("  Weights file: %s", weights_file);
    LOG("  Inputs file: %s", inputs_file);
    LOG("  Labels file: %s", labels_file);
    LOG("  Number of samples: %d", num_samples);
    LOG("  Batch size: %d", batch_size);
    
    // Using fixed point arithmetic
#if FIXED_POINT_ENABLE
    LOG("Using fixed-point arithmetic with %d-bit fractional part", 
        (int)log2(FIXED_POINT_SCALE));
#else
    LOG("Using floating-point arithmetic");
#endif
    
    // Create and initialize network
    Network *network = create_network(config_file, weights_file);
    if (!network) {
        ERR("Failed to create network");
        return 1;
    }
    
    // Print model summary
    print_model_summary(network);
    
    // Allocate memory for input and labels
    fixed_t *inputs = (fixed_t *)malloc(num_samples * network->input_size * sizeof(fixed_t));
    int *labels = (int *)malloc(num_samples * sizeof(int));
    
    if (!inputs || !labels) {
        ERR("Failed to allocate memory for inputs or labels");
        if (inputs) free(inputs);
        if (labels) free(labels);
        free_network(network);
        return 1;
    }
    
    // Load sample data
    load_samples(inputs, labels, num_samples, inputs_file, labels_file);
    
    // Run standard evaluation (one sample at a time)
    evaluate_model(network, inputs, labels, num_samples);
    
    // Run batch inference
    batch_inference(network, inputs, labels, num_samples, batch_size);
    
    // Cleanup
    free(inputs);
    free(labels);
    free_network(network);
    
    LOG("Program completed successfully");
    return 0;
}