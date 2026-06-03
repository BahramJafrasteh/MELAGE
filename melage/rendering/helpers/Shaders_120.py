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

    varying vec2 fragTexCoord;

    // --- INPUTS ---
    uniform sampler2D tex;
    uniform vec2 iResolution;

    // --- CONTROLS ---
    uniform float u_denoise;
    uniform float u_structure;
    
    // BACKPORTED: Added from 330
    uniform float u_isVideoRange;   
    uniform float u_glareComp;      

    // Color Grading
    uniform float u_brightness;     
    uniform float u_contrast;       
    uniform float u_saturation;     
    uniform float u_gamma;          
    uniform float u_nbi;

    const vec3 LUMA = vec3(0.2126, 0.7152, 0.0722); 

    // --- SORTING UTILS FOR MEDIAN FILTER ---
    #define s2(a, b) temp = a; a = min(a, b); b = max(temp, b);
    #define mn3(a, b, c) s2(a, b); s2(a, c); s2(b, c);
    #define mx3(a, b, c) s2(a, b); s2(a, c); s2(b, c);

    vec3 applyMedianFilter(sampler2D t, vec2 uv, vec2 px) {
        vec3 v[9];
        for(int dX = -1; dX <= 1; ++dX) {
            for(int dY = -1; dY <= 1; ++dY) {
                vec2 offset = vec2(float(dX), float(dY));
                // MUST keep using texture2D for 1.20
                v[(dX + 1) * 3 + (dY + 1)] = texture2D(t, uv + offset * px).rgb; 
            }
        }
        vec3 temp;
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
        float hemoglobin = dot(color, vec3(0.0, 0.6, 0.4));
        vec3 nbiColor = vec3(
            color.g * 1.2,
            color.b * 1.1,
            color.b * 1.5
        );
        return mix(color, nbiColor, u_nbi);
    }

    vec3 applyColorGrade(vec3 color) {
        vec3 res = color;

        // 1. Contrast — pivot at 0.5 so mid-grey stays unchanged
        float cont = (u_contrast == 0.0) ? 1.0 : u_contrast;
        res = (res - 0.5) * cont + 0.5;

        // 2. Brightness — clamped to keep values in [0,1]
        res = clamp(res + u_brightness, 0.0, 1.0);

        // 3. Gamma — skip the pow() when it would be a no-op
        float gam = (u_gamma == 0.0) ? 1.0 : u_gamma;
        if (gam != 1.0) res = pow(max(res, vec3(0.0)), vec3(1.0 / gam));

        // Final clamp — eliminates any floating-point overshoot
        return clamp(res, 0.0, 1.0);
    }

    void main(void) {
        vec2 px = (iResolution.x > 0.0) ? (1.0 / iResolution) : vec2(1.0/1024.0, 1.0/768.0);

        // BACKPORTED: Sample and Range Expansion
        vec3 c = texture2D(tex, fragTexCoord).rgb;
        if (u_isVideoRange > 0.5) {
            c = clamp((c - (16.0 / 255.0)) * (255.0 / 219.0), 0.0, 1.0);
        }

        vec3 blurred = applyMedianFilter(tex, fragTexCoord, px);
        if (u_isVideoRange > 0.5) {
             blurred = clamp((blurred - (16.0 / 255.0)) * (255.0 / 219.0), 0.0, 1.0);
        }
        vec3 base = mix(c, blurred, u_denoise);

        // 2. STRUCTURE (Unsharp Mask — 8-neighbour Laplacian)
        // Cardinal neighbours weighted x2, diagonals x1 (~Gaussian kernel, divisor 12).
        // Avoids the cross-shaped artefacts that appear with 4-neighbour kernels.
        vec3 n  = texture2D(tex, fragTexCoord + vec2( 0.0,  -px.y)).rgb;
        vec3 sv = texture2D(tex, fragTexCoord + vec2( 0.0,   px.y)).rgb;
        vec3 e  = texture2D(tex, fragTexCoord + vec2( px.x,  0.0 )).rgb;
        vec3 w  = texture2D(tex, fragTexCoord + vec2(-px.x,  0.0 )).rgb;
        vec3 ne = texture2D(tex, fragTexCoord + vec2( px.x, -px.y)).rgb;
        vec3 nw = texture2D(tex, fragTexCoord + vec2(-px.x, -px.y)).rgb;
        vec3 se = texture2D(tex, fragTexCoord + vec2( px.x,  px.y)).rgb;
        vec3 sw = texture2D(tex, fragTexCoord + vec2(-px.x,  px.y)).rgb;

        if (u_isVideoRange > 0.5) {
            float lo = 16.0 / 255.0;
            float sc = 255.0 / 219.0;
            n  = clamp((n  - lo) * sc, 0.0, 1.0);
            sv = clamp((sv - lo) * sc, 0.0, 1.0);
            e  = clamp((e  - lo) * sc, 0.0, 1.0);
            w  = clamp((w  - lo) * sc, 0.0, 1.0);
            ne = clamp((ne - lo) * sc, 0.0, 1.0);
            nw = clamp((nw - lo) * sc, 0.0, 1.0);
            se = clamp((se - lo) * sc, 0.0, 1.0);
            sw = clamp((sw - lo) * sc, 0.0, 1.0);
        }

        vec3 neighbors = (n + sv + e + w) * 2.0 + (ne + nw + se + sw);
        neighbors /= 12.0;
        vec3 detail = base - neighbors;
        vec3 structured = base + detail * u_structure;

        if (u_nbi > 0.0) {
            structured = applyNBI(structured);
        }

        // BACKPORTED: GLARE COMPRESSION with toggle
        if (u_glareComp > 0.5) {
            float lum = dot(structured, LUMA);
            if (lum > 0.8) {
                float compression = 1.0 - smoothstep(0.8, 1.2, lum) * 0.3;
                structured *= compression;
            }
        }

        vec3 finalColor = applyColorGrade(structured);

        // MUST keep using gl_FragColor for 1.20
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
