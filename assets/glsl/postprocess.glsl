#ifndef POSTPROCESS_GLSL
#define POSTPROCESS_GLSL

// From http://filmicgames.com/archives/75
vec3 Uncharted2Tonemap(vec3 x)
{
	float A = 0.15;
	float B = 0.50;
	float C = 0.10;
	float D = 0.20;
	float E = 0.02;
	float F = 0.30;
	return ((x*(A*x+C*B)+D*E)/(x*(A*x+B)+D*F))-E/F;
}

vec3 tonemapUncharted2( in vec3 color )
{
	const float W = 11.2;
	const float exposureBias = 2.0;
	vec3 curr = Uncharted2Tonemap(exposureBias * color);
	vec3 whiteScale = 1.0 / Uncharted2Tonemap(vec3(W));
	return curr * whiteScale;
}

vec3 ACESFilm( vec3 x ) {
    float a = 2.51f;
    float b = 0.03f;
    float c = 2.43f;
    float d = 0.59f;
    float e = 0.14f;
    return clamp( ( x * ( a * x + b ) ) / ( x * ( c * x + d ) + e ), vec3(0), vec3(1) );
}

vec3 exposure(vec3 color, float fstop) {
   return color * pow(2.0,fstop);
}

vec3 gammaCorrect( in vec3 color, float power )
{
    return pow( color, vec3(1.0f / power) );
}

#endif