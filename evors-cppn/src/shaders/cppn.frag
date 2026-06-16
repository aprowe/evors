#version 430
// Generic CPPN interpreter. Walks a topologically-ordered node list (slots
// 0..u_node_count) and its incoming connections, both uploaded as SSBOs, to
// evaluate the network per pixel. Inputs occupy slots 0..3: x, y, d, bias.

out vec4 frag;

// nodes[s] = (activation_id, conn_start, conn_count, _pad)
layout(std430, binding = 0) readonly buffer Nodes { uvec4 nodes[]; };
// conns[i] = (source_slot, floatBitsToUint(weight))
layout(std430, binding = 1) readonly buffer Conns { uvec2 conns[]; };

uniform uint u_node_count;
uniform uvec3 u_out_slots;
uniform vec2 u_res;
// Output alpha. 1.0 for opaque eval renders; < 1.0 for the display pass, where
// alpha blending into an accumulation buffer produces a smooth temporal average.
uniform float u_alpha;

const int MAX_NODES = 256;

float activation(uint id, float x) {
    if (id == 0u) return x;                       // Identity
    else if (id == 1u) return 1.0 / (1.0 + exp(-4.9 * x)); // Sigmoid
    else if (id == 2u) return tanh(x);            // Tanh
    else if (id == 3u) return sin(x);             // Sin
    else if (id == 4u) return exp(-x * x);        // Gauss
    else return abs(x);                           // Abs
}

void main() {
    float value[MAX_NODES];

    vec2 xy = (gl_FragCoord.xy / u_res) * 2.0 - 1.0;
    value[0] = xy.x;
    value[1] = xy.y;
    value[2] = length(xy);
    value[3] = 1.0;

    uint n = min(u_node_count, uint(MAX_NODES));
    for (uint s = 4u; s < n; s++) {
        uvec4 node = nodes[s];
        float acc = 0.0;
        for (uint k = 0u; k < node.z; k++) {
            uvec2 c = conns[node.y + k];
            acc += value[c.x] * uintBitsToFloat(c.y);
        }
        value[s] = activation(node.x, acc);
    }

    frag = vec4(
        clamp(value[u_out_slots.x], 0.0, 1.0),
        clamp(value[u_out_slots.y], 0.0, 1.0),
        clamp(value[u_out_slots.z], 0.0, 1.0),
        u_alpha
    );
}
