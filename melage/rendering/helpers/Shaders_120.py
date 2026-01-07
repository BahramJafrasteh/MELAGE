import textwrap

__AUTHOR__ = 'Bahram Jafrasteh'

# ---------------------------------------------------------
# Vertex Shader (Main)
# ---------------------------------------------------------
vsrc = """
#version 120

// 'in' becomes 'attribute' in 1.20
attribute vec2 in_Vertex;
attribute vec2 vertTexCoord;

// 'out' becomes 'varying'
varying vec2 fragTexCoord;

uniform mat4 g_matModelView;

void main(void) {
    // REVERTED: Kept your original multiplication order
    // (Vector * Matrix) handles row-major matrices correctly
    gl_Position = vec4(in_Vertex, 0.0, 1.0) * g_matModelView;

    fragTexCoord = vertTexCoord;
}
"""

# ---------------------------------------------------------
# Fragment Shader (Image Processing)
# ---------------------------------------------------------
fsrc = textwrap.dedent("""
    #version 120

    // 'in' becomes 'varying'
    varying vec2 fragTexCoord;

    // GLSL 1.20 writes to gl_FragColor automatically, no 'out' needed

    // --- INPUTS ---
    uniform sampler2D tex;
    uniform vec2 iResolution;

    // --- CONTROLS ---
    uniform float u_denoise;
    uniform float u_structure;

    // Color Grading
    uniform float u_brightness;     
    uniform float u_contrast;       
    uniform float u_saturation;     
    uniform float u_gamma;          

    uniform float u_nbi;

    // Standard Luma
    const vec3 LUMA = vec3(0.2126, 0.7152, 0.0722); 

    // --- SORTING UTILS FOR MEDIAN FILTER ---
    #define s2(a, b) temp = a; a = min(a, b); b = max(temp, b);
    #define mn3(a, b, c) s2(a, b); s2(a, c); s2(b, c);
    #define mx3(a, b, c) s2(a, b); s2(a, c); s2(b, c);

    // High-performance 3x3 Median 
    vec3 applyMedianFilter(sampler2D t, vec2 uv, vec2 px) {
        vec3 v[9];

        // 1. Sample the 3x3 grid
        // Note: texture() becomes texture2D() in 1.20
        for(int dX = -1; dX <= 1; ++dX) {
            for(int dY = -1; dY <= 1; ++dY) {
                vec2 offset = vec2(float(dX), float(dY));
                v[(dX + 1) * 3 + (dY + 1)] = texture2D(t, uv + offset * px).rgb;
            }
        }

        vec3 temp;

        // 2. Optimized Sorting Network for 9 elements
        s2(v[1], v[2]); s2(v[4], v[5]); s2(v[7], v[8]);
        s2(v[0], v[1]); s2(v[3], v[4]); s2(v[6], v[7]);
        s2(v[1], v[2]); s2(v[4], v[5]); s2(v[7], v[8]);
        s2(v[0], v[3]); s2(v[5], v[8]); s2(v[4], v[7]);
        s2(v[3], v[6]); s2(v[1], v[4]); s2(v[2], v[5]);
        s2(v[4], v[7]); s2(v[4], v[2]); s2(v[6], v[4]);
        s2(v[4], v[2]);

        return v[4];
    }

    vec3 applyNBI(vec3 color) {
        // 1. Grayscale extraction heavily weighted to blue/green
        float hemoglobin = dot(color, vec3(0.0, 0.6, 0.4));

        // 2. Create a "false color" map
        vec3 nbiColor = vec3(
            color.g * 1.2,
            color.b * 1.1,
            color.b * 1.5
        );

        // 3. Mix based on intensity
        return mix(color, nbiColor, u_nbi);
    }

    // --- COLOR GRADING HELPER ---
    vec3 applyColorGrade(vec3 color) {
        vec3 res = color;

        // 1. Saturation
        float intensity = dot(res, LUMA);
        float sat = 1.0; 
        res = mix(vec3(intensity), res, sat);

        // 2. Contrast
        float cont = (u_contrast == 0.0) ? 1.0 : u_contrast;
        res = (res - 0.5) * cont + 0.5;

        // 3. Brightness
        res = res + u_brightness;

        // 4. Gamma
        float gam = (u_gamma == 0.0) ? 1.0 : u_gamma;
        if (gam > 0.0) res = pow(max(res, vec3(0.0)), vec3(1.0 / gam));

        return res;
    }

    void main(void) {
        vec2 px = (iResolution.x > 0.0) ? (1.0 / iResolution) : vec2(1.0/1024.0, 1.0/768.0);

        // ------------------------------------------------
        // 1. DENOISE
        // ------------------------------------------------
        vec3 c  = texture2D(tex, fragTexCoord).rgb;
        vec3 blurred = applyMedianFilter(tex, fragTexCoord, px);
        vec3 base = mix(c, blurred, u_denoise);

        // ------------------------------------------------
        // 2. STRUCTURE 
        // ------------------------------------------------
        vec3 n = texture2D(tex, fragTexCoord + vec2(0.0, -px.y)).rgb;
        vec3 s = texture2D(tex, fragTexCoord + vec2(0.0, px.y)).rgb;
        vec3 e = texture2D(tex, fragTexCoord + vec2(px.x, 0.0)).rgb;
        vec3 w = texture2D(tex, fragTexCoord + vec2(-px.x, 0.0)).rgb;

        vec3 neighbors = (n + s + e + w) * 0.25;
        vec3 detail = base - neighbors;
        vec3 structured = base + detail * u_structure;

        // 3. VASCULAR ENHANCEMENT (NBI)
        if (u_nbi > 0.0) {
            structured = applyNBI(structured);
        }

        // ------------------------------------------------
        // 3. GLARE COMPRESSION
        // ------------------------------------------------
        float lum = dot(structured, LUMA);
        if (lum > 0.8) {
            float compression = 1.0 - smoothstep(0.8, 1.2, lum) * 0.3;
            structured *= compression;
        }

        // ------------------------------------------------
        // 4. FINAL COLOR GRADING
        // ------------------------------------------------
        vec3 finalColor = applyColorGrade(structured);

        gl_FragColor = vec4(finalColor, 1.0);
    }
""")

# ---------------------------------------------------------
# Vertex Shader (Paint/Points)
# ---------------------------------------------------------
vsrcPaint = textwrap.dedent("""
    #version 120

    attribute vec2 in_Vertex;
    uniform mat4 g_matModelView;

    void main(void) {
        gl_Position =  vec4(in_Vertex, 0.0, 1.0)*g_matModelView;
        gl_PointSize = 200.0; 
    }
""")

# ---------------------------------------------------------
# Fragment Shader (Paint/Points)
# ---------------------------------------------------------
fsrcPaint = textwrap.dedent("""
    #version 120

    uniform vec4 my_color;

    void main(void) {
        gl_FragColor = my_color;
    }
""")