

__AUTHOR__ = 'Bahram Jafrasteh'


#vTexCoord=[0.0,0.0,  1.0,0.0,  1.0,1.0,  0.0,1.0]
#vVertices= [-1.0,-1.0,  1.0,-1.0,  1.0,1.0,  -1.0,1.0]

#self.coord = [(0, 0), (0, 1), (1, 1), (1, 0)]
#self.vertex = [(0, 0), (0, self.imHeight), (self.imWidth, self.imHeight), (self.imWidth, 0)]

vsrc = """
#version 120

attribute vec2 in_Vertex;
attribute vec2 vertTexCoord;
varying vec2 fragTexCoord;
uniform mat4 g_matModelView;

void main(void)
{
    gl_Position = vec4(in_Vertex, 0.0, 1.0) * g_matModelView;
    fragTexCoord = vertTexCoord;
}
"""


fsrc = """
#version 120

const float PI = 3.14159265359;
uniform float u_transformSize;
uniform float u_subtransformSize;

uniform sampler2D u_input;
uniform sampler2D tex;
uniform float threshold;
uniform float contrastMult;
uniform float brightnessAdd;
uniform vec3 deinterlace;
uniform vec2 mousePos;
uniform vec2 iResolution;
uniform float maxRadius;
uniform float Ilum;

const float fZero= 0.0;
const float fOne= 1.0;
const float fTwo= 2.0;
const float fColScal= 256.0;
const float Epsilon= 1e-10;

varying vec2 fragTexCoord;

uniform int sobel;
uniform float sobel_threshold;

vec2 multiplyComplex (vec2 a, vec2 b) {
    return vec2(a[0] * b[0] - a[1] * b[1], a[1] * b[0] + a[0] * b[1]);
}

vec4 fft(vec2 vUV, float u_transformSize, float u_subtransformSize){
    float index = vUV.x * u_transformSize - 0.5;
    float evenIndex = floor(index / u_subtransformSize) * (u_subtransformSize * 0.5) + mod(index, u_subtransformSize * 0.5);

    vec4 even = texture2D(u_input, vec2(evenIndex + 0.5, gl_FragCoord.y) / u_transformSize);
    vec4 odd = texture2D(u_input, vec2(evenIndex + u_transformSize * 0.5 + 0.5, gl_FragCoord.y) / u_transformSize);

    float twiddleArgument1D = -2.0 * PI * (index / u_subtransformSize);
    vec2 twiddle1D = vec2(cos(twiddleArgument1D), sin(twiddleArgument1D));

    vec2 outputA = even.xy + multiplyComplex(twiddle1D, odd.xy);
    vec2 outputB = even.zw + multiplyComplex(twiddle1D, odd.zw);

    return vec4(outputA, outputB);
}

vec4 sobel_kernel(sampler2D tex, vec2 coord)
{
    float w = maxRadius / iResolution.x;
    float h = maxRadius / iResolution.y;

    vec3 BL = texture2D(tex, coord + vec2(-w, -h)).rgb;
    vec3 BM = texture2D(tex, coord + vec2(0.0, -h)).rgb;
    vec3 BR = texture2D(tex, coord + vec2(w, -h)).rgb;
    vec3 ML = texture2D(tex, coord + vec2(-w, 0.0)).rgb;
    vec3 MM = texture2D(tex, coord).rgb;
    vec3 MR = texture2D(tex, coord + vec2(w, 0.0)).rgb;
    vec3 TL = texture2D(tex, coord + vec2(-w, h)).rgb;
    vec3 TM = texture2D(tex, coord + vec2(0.0, h)).rgb;
    vec3 TR = texture2D(tex, coord + vec2(w, h)).rgb;
    vec3 GradX = -TL + TR - 2.0 * ML + 2.0 * MR - BL + BR;
    vec3 GradY = TL + 2.0 * TM + TR - BL - 2.0 * BM - BR;
    return vec4(length(vec2(GradX.r, GradY.r)), length(vec2(GradX.g, GradY.g)), length(vec2(GradX.b, GradY.b)), 1);
}

vec3 RGBtoHSV(in vec3 RGB)
{
    vec4 P = (RGB.g < RGB.b) ? vec4(RGB.bg, -1.0, 2.0 / 3.0) : vec4(RGB.gb, 0.0, -1.0 / 3.0);
    vec4 Q = (RGB.r < P.x) ? vec4(P.xyw, RGB.r) : vec4(RGB.r, P.yzx);
    float C = Q.x - min(Q.w, Q.y);
    float H = abs((Q.w - Q.y) / (6.0 * C + Epsilon) + Q.z);
    vec3 HCV = vec3(H, C, Q.x);
    float S = HCV.y / (HCV.z + Epsilon);
    return vec3(HCV.x, S, HCV.z);
}

vec3 HSVtoRGB(in vec3 HSV)
{
    float H = HSV.x;
    float R = abs(H * 6.0 - 3.0) - 1.0;
    float G = 2.0 - abs(H * 6.0 - 2.0);
    float B = 2.0 - abs(H * 6.0 - 4.0);
    vec3 RGB = clamp(vec3(R, G, B), 0.0, 1.0);
    return ((RGB - 1.0) * HSV.y + 1.0) * HSV.z;
}

mat3 kernel1 = mat3(1.0, 0.0, -1.0,
                    2.0, 0.0, 2.0,
                    1.0, 0.0, -1.0);
mat3 kernel2 = mat3(1.0, 2.0, 1.0,
                    0.0, 0.0, 0.0,
                    -1.0, -2.0, -1.0);

float toGrayscale(vec3 source)
{
    float average = (source.x + source.y + source.z) / 3.0;
    return average;
}

float doConvolution(mat3 kernel)
{
    float sum = 0.0;
    float current_pixelColor = toGrayscale(texture2D(tex, fragTexCoord).xyz);
    float xOffset = 1.0 / 1024.0;
    float yOffset = 1.0 / 768.0;
    float new_pixel00 = toGrayscale(texture2D(tex, vec2(fragTexCoord.x - xOffset, fragTexCoord.y - yOffset)).xyz);
    float new_pixel01 = toGrayscale(texture2D(tex, vec2(fragTexCoord.x, fragTexCoord.y - yOffset)).xyz);
    float new_pixel02 = toGrayscale(texture2D(tex, vec2(fragTexCoord.x + xOffset, fragTexCoord.y - yOffset)).xyz);
    vec3 pixelRow0 = vec3(new_pixel00, new_pixel01, new_pixel02);
    float new_pixel10 = toGrayscale(texture2D(tex, vec2(fragTexCoord.x - xOffset, fragTexCoord.y)).xyz);
    float new_pixel11 = toGrayscale(texture2D(tex, vec2(fragTexCoord.x, fragTexCoord.y)).xyz);
    float new_pixel12 = toGrayscale(texture2D(tex, vec2(fragTexCoord.x + xOffset, fragTexCoord.y)).xyz);
    vec3 pixelRow1 = vec3(new_pixel10, new_pixel11, new_pixel12);
    float new_pixel20 = toGrayscale(texture2D(tex, vec2(fragTexCoord.x - xOffset, fragTexCoord.y + yOffset)).xyz);
    float new_pixel21 = toGrayscale(texture2D(tex, vec2(fragTexCoord.x, fragTexCoord.y + yOffset)).xyz);
    float new_pixel22 = toGrayscale(texture2D(tex, vec2(fragTexCoord.x + xOffset, fragTexCoord.y + yOffset)).xyz);
    vec3 pixelRow2 = vec3(new_pixel20, new_pixel21, new_pixel22);
    vec3 mult1 = (kernel[0] * pixelRow0);
    vec3 mult2 = (kernel[1] * pixelRow1);
    vec3 mult3 = (kernel[2] * pixelRow2);
    sum = mult1.x + mult1.y + mult1.z + mult2.x + mult2.y + mult2.z + mult3.x + mult3.y + mult3.z;
    return sum;
}

void main(void)
{
    int gaussianBlue = 0;

    if (sobel == 1) //sobel kernel
    {
        vec4 color0 = texture2D(tex, fragTexCoord);
        vec4 bw = vec4(vec3(color0.r, color0.g, color0.b), color0.a);
        vec2 uv = gl_FragCoord.xy / iResolution.xy;
        vec2 mousePosition = mousePos / iResolution.xy;
        float dist = distance(mousePosition, uv);
        float distx = abs(mousePosition.x - uv.x);
        float disty = abs(mousePosition.y - uv.y);
        float mixAmount = clamp((dist - 0.0) / 0.2, 0., 1.);

        vec4 sobelk = sobel_kernel(tex, fragTexCoord.st);

        float luminance = dot(vec3(0.114, 0.587, 0.299), color0.rgb);

        if (luminance < sobel_threshold)
            gl_FragColor = vec4(color0.rgba);
        else
        {
            color0.r += sobelk.r;
            color0.g = sobelk.g;
            color0.b += sobelk.b;
        }

        gl_FragColor = mix(color0, bw, mixAmount);

        vec4 pixelColor = texture2D(tex, fragTexCoord);
        float horizontalSum = 0.0;
        float verticalSum = 0.0;
        float averageSum = 0.0;
        horizontalSum = doConvolution(kernel1);
        verticalSum = doConvolution(kernel2);
        if ((verticalSum > sobel_threshold) || (horizontalSum > sobel_threshold) || (verticalSum < -sobel_threshold) || (horizontalSum < -sobel_threshold))
            averageSum = pixelColor.x;
        else
            averageSum = 1.0;
        gl_FragColor = vec4(averageSum, averageSum, averageSum, 1.0);
    }
    else
    {
        vec4 pixelColor = texture2D(tex, fragTexCoord);

        pixelColor.rgb /= pixelColor.a;

        // first apply brightness
        pixelColor.rgb += brightnessAdd;

        // then apply contrast
        pixelColor.rgb = ((pixelColor.rgb - 0.5) * max(contrastMult, 0.0)) + 0.5;

        // Return final pixel color
        pixelColor.rgb *= pixelColor.a;

        vec3 current_Color;

        current_Color = vec3(pixelColor.r, pixelColor.g, pixelColor.b);

        if (threshold > 0)
        {
            float luminance = dot(vec3(0.114, 0.587, 0.299), current_Color);
            if (luminance < threshold)
                gl_FragColor = vec4(1.0);
            else
                gl_FragColor = vec4(0.0);
        }
        else
        {
            gl_FragColor = vec4(current_Color, 1.0);
        }
    }
}
"""

vsrcPaint = """
#version 120

attribute vec2 in_Vertex;
uniform mat4 g_matModelView;
varying vec4 positionGL;

void main(void)
{
    gl_Position =  vec4(in_Vertex, 0.0, 1.0)*g_matModelView;
    positionGL = gl_Position;
    gl_PointSize = 200.0;
}
"""

fsrcPaint = """
#version 120

uniform vec4 my_color;
varying vec4 positionGL;

void main(void)
{
    gl_FragColor = my_color;
}
"""
