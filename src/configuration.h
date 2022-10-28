#ifndef __whak
#define __whak

#include "al/app/al_App.hpp"
// #include "al/app/al_DistributedApp.hpp"
#include "al/ui/al_ControlGUI.hpp"
#include "al/ui/al_Parameter.hpp"
#include "al/ui/al_PresetHandler.hpp"
#include "al/ui/al_PresetServer.hpp"
#include "al/sound/al_Reverb.hpp"
// #include "al_ext/assets3d/al_Asset.hpp"
// #include "al_ext/statedistribution/al_CuttleboneStateSimulationDomain.hpp"

#include "synths.h"
#include "Gamma/Noise.h"
#include "Gamma/Delay.h"
#include "Gamma/Oscillator.h"
#include "Gamma/Filter.h"
#include "Gamma/Envelope.h"
#include "Gamma/Effects.h"
#include "Gamma/Analysis.h"
#include "Gamma/DFT.h"
#include <forward_list>
#include <string>
#include <unordered_set>
#include <vector>

namespace ben {
    const int window_size = 128;
    const int hopSize = 32;
    const int N = 1920;
    const float CLOUD_WIDTH = 1.0;

    const char* vertex = R"(
    #version 400

    layout (location = 0) in vec3 vertexPosition;
    layout (location = 1) in vec4 vertexColor;

    uniform mat4 al_ModelViewMatrix;
    uniform mat4 al_ProjectionMatrix;

    out Vertex {
    vec4 color;
    } vertex;

    void main() {
    gl_Position = al_ModelViewMatrix * vec4(vertexPosition, 1.0);
    vertex.color = vertexColor;
    }
    )";

    const char* fragment = R"(
    #version 400

    in Fragment {
    vec4 color;
    vec2 textureCoordinate;
    } fragment;

    uniform sampler2D alphaTexture;

    layout (location = 0) out vec4 fragmentColor;

    void main() {
    // use the first 3 components of the color (xyz is rgb), but take the alpha value from the texture
    //
    fragmentColor = vec4(fragment.color.xyz, texture(alphaTexture, fragment.textureCoordinate));
    }
    )";

    const char* geometry = R"(
    #version 400

    // take in a point and output a triangle strip with 4 vertices (aka a "quad")
    //
    layout (points) in;
    layout (triangle_strip, max_vertices = 4) out;

    uniform mat4 al_ProjectionMatrix;

    // this uniform is *not* passed in automatically by AlloLib; do it manually
    //
    uniform float halfSize;

    in Vertex {
    vec4 color;
    } vertex[];

    out Fragment {
    vec4 color;
    vec2 textureCoordinate;
    } fragment;

    void main() {
    mat4 m = al_ProjectionMatrix; // rename to make lines shorter
    vec4 v = gl_in[0].gl_Position; // al_ModelViewMatrix * gl_Position

    gl_Position = m * (v + vec4(-halfSize, -halfSize, 0.0, 0.0));
    fragment.textureCoordinate = vec2(0.0, 0.0);
    fragment.color = vertex[0].color;
    EmitVertex();

    gl_Position = m * (v + vec4(halfSize, -halfSize, 0.0, 0.0));
    fragment.textureCoordinate = vec2(1.0, 0.0);
    fragment.color = vertex[0].color;
    EmitVertex();

    gl_Position = m * (v + vec4(-halfSize, halfSize, 0.0, 0.0));
    fragment.textureCoordinate = vec2(0.0, 1.0);
    fragment.color = vertex[0].color;
    EmitVertex();

    gl_Position = m * (v + vec4(halfSize, halfSize, 0.0, 0.0));
    fragment.textureCoordinate = vec2(1.0, 1.0);
    fragment.color = vertex[0].color;
    EmitVertex();

    EndPrimitive();
    }
    )";
}  // namespace diy
#endif  