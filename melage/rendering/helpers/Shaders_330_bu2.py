import textwrap

__AUTHOR__ = 'Bahram Jafrasteh'

# ---------------------------------------------------------
# Vertex Shader (Main)
# ---------------------------------------------------------
vsrc = """
#version 330 core

layout(location = 0) in vec2 in_Vertex;
layout(location = 1) in vec2 vertTexCoord;

out vec2 fragTexCoord;

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
import textwrap

import textwrap

import textwrap

fsrc = textwrap.dedent("""
    #version 330 core

    in vec2 fragTexCoord;
    out vec4 fragColor;

    // --- INPUTS ---
    uniform sampler2D tex;
    uniform vec2 iResolution;

    // --- CONTROLS ---
    // 0.0 = Off, 1.0 = Max Effect
    uniform float u_denoise;    // Smoothing (reduces grain)
    uniform float u_structure;  // Gentle detail boost (better than sharpen)

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

#define mnmx4(a, b, c, d) s2(a, b); s2(c, d); s2(a, c); s2(b, d);
#define mnmx5(a, b, c, d, e) s2(a, b); s2(c, d); mn3(a, c, e); mx3(b, d, e);
#define mnmx6(a, b, c, d, e, f) s2(a, d); s2(b, e); s2(c, f); mn3(a, b, c); mx3(d, e, f);

// High-performance 3x3 Median 
// This effectively removes the "black grid" from fiber scopes
vec3 applyMedianFilter(sampler2D t, vec2 uv, vec2 px) {
    vec3 v[9];

    // 1. Sample the 3x3 grid
    for(int dX = -1; dX <= 1; ++dX) {
        for(int dY = -1; dY <= 1; ++dY) {
            vec2 offset = vec2(float(dX), float(dY));
            v[(dX + 1) * 3 + (dY + 1)] = texture(t, uv + offset * px).rgb;
        }
    }

    vec3 temp;

    // 2. Optimized Sorting Network for 9 elements
    // This sorts the pixel array to find the true "middle" color
    s2(v[1], v[2]); s2(v[4], v[5]); s2(v[7], v[8]);
    s2(v[0], v[1]); s2(v[3], v[4]); s2(v[6], v[7]);
    s2(v[1], v[2]); s2(v[4], v[5]); s2(v[7], v[8]);
    s2(v[0], v[3]); s2(v[5], v[8]); s2(v[4], v[7]);
    s2(v[3], v[6]); s2(v[1], v[4]); s2(v[2], v[5]);
    s2(v[4], v[7]); s2(v[4], v[2]); s2(v[6], v[4]);
    s2(v[4], v[2]);
    
    // v[4] is now the median value
    return v[4];
}
    
    
    vec3 applyNBI(vec3 color) {
        // Physical NBI uses 415nm (Blue) and 540nm (Green) light.
        // Digital approximation: Drop Red, map Green->Red, Blue->Green+Blue
        
        // 1. Grayscale extraction heavily weighted to blue/green
        float hemoglobin = dot(color, vec3(0.0, 0.6, 0.4));
        
        // 2. Create a "false color" map
        vec3 nbiColor = vec3(
            color.g * 1.2,  // Map Green channel to Red output (enhances surface vessels)
            color.b * 1.1,  // Map Blue to Green
            color.b * 1.5   // Boost Blue for deep contrast
        );
    
        // 3. Mix based on intensity to keep it viewable
        return mix(color, nbiColor, u_nbi);
    }

    // --- COLOR GRADING HELPER ---
    vec3 applyColorGrade(vec3 color) {
        vec3 res = color;

        // 1. Saturation
        float intensity = dot(res, LUMA);
        float sat = 1.0; 
        // You can link this to a uniform if you wish
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
        // 1. DENOISE (Gaussian Smoothing)
        // ------------------------------------------------
        // We sample the center and 4 neighbors to create a smooth base.
        vec3 c  = texture(tex, fragTexCoord).rgb;
        //vec3 n  = texture(tex, fragTexCoord + vec2( 0.0, -px.y)).rgb;
        //vec3 s  = texture(tex, fragTexCoord + vec2( 0.0,  px.y)).rgb;
        //vec3 e  = texture(tex, fragTexCoord + vec2( px.x,  0.0)).rgb;
        //vec3 w  = texture(tex, fragTexCoord + vec2(-px.x,  0.0)).rgb;

        // Calculate a blurred version
        //vec3 blurred = (c * 4.0 + n + s + e + w) * 0.125;
        vec3 blurred = applyMedianFilter(tex, fragTexCoord, px);

        // Mix: If u_denoise is 0.0, we use original 'c'. 
        // If u_denoise is 1.0, we use 'blurred'.
        // Suggested default for Endoscopy: 0.3
        vec3 base = mix(c, blurred, u_denoise);

        // ------------------------------------------------
        // 2. STRUCTURE (Unsharp Mask on the SMOOTH base)
        // ------------------------------------------------
        // We calculate detail using the *smoothed* image vs the *blurred* image.
        // This prevents us from sharpening the noise (grain).
        // Simple neighbor check for edges
        vec3 n = texture(tex, fragTexCoord + vec2(0.0, -px.y)).rgb;
        vec3 s = texture(tex, fragTexCoord + vec2(0.0, px.y)).rgb;
        vec3 e = texture(tex, fragTexCoord + vec2(px.x, 0.0)).rgb;
        vec3 w = texture(tex, fragTexCoord + vec2(-px.x, 0.0)).rgb;
        
        vec3 neighbors = (n + s + e + w) * 0.25;
        
        vec3 detail = base - neighbors;

        // Apply detail. 
        // u_structure: 0.0 = Flat, 2.0 = High Definition
        // Suggested default: 0.5
        vec3 structured = base + detail * u_structure;

        
        // 3. VASCULAR ENHANCEMENT (NBI)
            if (u_nbi > 0.0) {
                structured = applyNBI(structured);
            }

        // ------------------------------------------------
        // 3. GLARE COMPRESSION (Tone Mapping)
        // ------------------------------------------------
        // Instead of painting over glare, we dampen the brightest pixels.
        // This recovers texture in wet areas without artifacts.
        float lum = dot(structured, LUMA);
        if (lum > 0.8) {
            // Softly compress highlights above 0.8 luminance
            float compression = 1.0 - smoothstep(0.8, 1.2, lum) * 0.3;
            structured *= compression;
        }

        

        // ------------------------------------------------
        // 4. FINAL COLOR GRADING
        // ------------------------------------------------
        vec3 finalColor = applyColorGrade(structured);

        fragColor = vec4(finalColor, 1.0);
    }
""")
# ---------------------------------------------------------
# Vertex Shader (Paint/Points)
# ---------------------------------------------------------
vsrcPaint = textwrap.dedent("""
    #version 330 core

    in vec2 in_Vertex;
    uniform mat4 g_matModelView;

    void main(void) {
        gl_Position = g_matModelView * vec4(in_Vertex, 0.0, 1.0);

        // Point size requires GL_PROGRAM_POINT_SIZE enabled in host code
        gl_PointSize = 200.0; 
    }
""")

# ---------------------------------------------------------
# Fragment Shader (Paint/Points)
# ---------------------------------------------------------
fsrcPaint = textwrap.dedent("""
    #version 330 core

    uniform vec4 my_color;
    out vec4 FragColor;

    void main(void) {
        FragColor = my_color;
    }
""")