#ifndef COMPLEX_H
#define COMPLEX_H

typedef struct {
	float re;
	float im;
} m_complex;

typedef struct {
	m_complex x;
	m_complex y;
	m_complex z;
} complex3;

__host__ __device__ __inline__ float abs(m_complex c)
{
	return sqrtf(c.re*c.re + c.im*c.im);
}

__host__ __device__ __inline__ float abs_sqr(m_complex c)
{
  return c.re*c.re + c.im*c.im;
}


__host__ __device__ __inline__ m_complex conj(m_complex c)
{
	m_complex result = { c.re, -c.im };
	return result;
}

__host__ __device__ __inline__ m_complex operator+(m_complex c1, m_complex c2)
{
	m_complex result = { c1.re + c2.re, c1.im + c2.im };
	return result;
}

__host__ __device__ __inline__ m_complex operator+(m_complex c1, float c2)
{
	m_complex result = { c1.re + c2, c1.im };
	return result;
}

__host__ __device__ __inline__ m_complex operator+(float c1, m_complex c2)
{
	m_complex result = { c1 + c2.re, c2.im };
	return result;

}

__host__ __device__ __inline__ m_complex operator-(m_complex c)
{
  m_complex result = { -c.re, -c.im };
  return result;
}

__host__ __device__ __inline__ m_complex operator-(m_complex c1, m_complex c2)
{
	m_complex result = { c1.re - c2.re, c1.im - c2.im };
	return result;
}

__host__ __device__ __inline__ m_complex operator-(m_complex c1, float c2)
{
	m_complex result = { c1.re - c2, c1.im };
	return result;
}

__host__ __device__ __inline__ m_complex operator-(float c1, m_complex c2)
{
	m_complex result = { c1 - c2.re, -c2.im };
	return result;
}

__host__ __device__ __inline__ m_complex operator*(m_complex c1, m_complex c2)
{
	m_complex result = { c1.re*c2.re - c1.im*c2.im, c1.re*c2.im + c2.re*c1.im };
	return result;
}

__host__ __device__ __inline__ m_complex operator*(m_complex c1, float c2)
{
	m_complex result = { c1.re*c2, c1.im*c2 };
	return result;
}

__host__ __device__ __inline__ m_complex operator*(float c1, m_complex c2)
{
	m_complex result = { c1*c2.re, c1*c2.im };
	return result;
}

__host__ __device__ __inline__ m_complex operator/(m_complex c1, float c2)
{
	return c1*(1.0f/c2);
}

__host__ __device__ __inline__ m_complex operator/(m_complex c1, m_complex c2)
{
	return c1*conj(c2)/abs_sqr(c2);
}

__host__ __device__ __inline__ m_complex operator/(float c1, m_complex c2)
{
	return c1*conj(c2)/abs_sqr(c2);
}

__host__ __device__ __inline__ float phase(m_complex c)
{
  return atan2(c.im, c.re);
}

__host__ __device__ __inline__ m_complex pow(m_complex c, int n)
{
	float rn = static_cast<float>(pow(abs(c), n));
	float theta = phase(c)*static_cast<float>(n);
	m_complex result = { rn*cosf(theta), rn*sinf(theta) };
	return result;
}

__host__ __device__ __inline__ m_complex pow(m_complex c, float n)
{
  float rn = static_cast<float>(pow(abs(c), n));
  float theta = phase(c)*n;
  m_complex result = { rn*cosf(theta), rn*sinf(theta) };
  return result;
}

__host__ __device__ __inline__ m_complex sqrt(m_complex c)
{
	float r = abs(c);
  m_complex result = { sqrtf((r + c.re)*0.5f), copysignf(sqrtf(fmaxf((r - c.re), 0.0f)*0.5f), c.im) };
	return result;
}

__host__ __device__ __inline__ m_complex exp(m_complex c)
{
  float atten = expf(c.re);
  m_complex oscil = { cosf(c.im), sinf(c.im) };
  return oscil*atten;
}

__host__ __device__ __inline__ float3 abs(complex3 c)
{
	float3 result;
	result.x = abs(c.x);
	result.y = abs(c.y);
	result.z = abs(c.z);
	return result;
}

__host__ __device__ __inline__ complex3 conj(complex3 c)
{

	complex3 result = { conj(c.x), conj(c.y), conj(c.z) };
	return result;
}

__host__ __device__ __inline__ complex3 operator+(complex3 c1, complex3 c2)
{
	complex3 result = { c1.x + c2.x, c1.y + c2.y, c1.z + c2.z };
	return result;
}

__host__ __device__ __inline__ complex3 operator+(complex3 c1, float c2)
{
	complex3 result = { c1.x + c2, c1.y + c2, c1.z + c2};
	return result;
}

__host__ __device__ __inline__ complex3 operator+(float c1, complex3 c2)
{
	complex3 result = { c1 + c2.x, c1 + c2.y, c1 + c2.z };
	return result;

}

__host__ __device__ __inline__ complex3 operator-(complex3 c)
{
  complex3 result = { -c.x, -c.y, -c.z };
  return result;
}

__host__ __device__ __inline__ complex3 operator-(complex3 c1, complex3 c2)
{
	complex3 result = { c1.x - c2.x, c1.y - c2.y, c1.z - c2.z };
	return result;
}

__host__ __device__ __inline__ complex3 operator-(complex3 c1, float c2)
{
	complex3 result = { c1.x - c2, c1.y - c2, c1.z - c2 };
	return result;
}

__host__ __device__ __inline__ complex3 operator-(float c1, complex3 c2)
{
	complex3 result = { c1 - c2.x, c1 - c2.y, c1 - c2.z };
	return result;
}

__host__ __device__ __inline__ complex3 operator*(complex3 c1, complex3 c2)
{

	complex3 result = { c1.x*c2.x, c1.y*c2.y, c1.z*c2.z };
	return result;
}

__host__ __device__ __inline__ complex3 operator*(complex3 c1, float c2)
{
	complex3 result = { c1.x*c2, c1.y*c2, c1.z*c2};
	return result;
}

__host__ __device__ __inline__ complex3 operator*(float c1, complex3 c2)
{
	complex3 result = { c1*c2.x, c1*c2.y, c1*c2.z };
	return result;
}

__host__ __device__ __inline__ complex3 operator/(complex3 c1, float c2)
{
	return c1*(1.0f/c2);
}

__host__ __device__ __inline__ complex3 operator/(complex3 c1, complex3 c2)
{
	complex3 result = { c1.x/c2.x, c1.y/c2.y, c1.z/c2.z };
	return result;
}

__host__ __device__ __inline__ complex3 operator/(float c1, complex3 c2)
{
	complex3 result = { c1/c2.x, c1/c2.y, c1/c2.z };
	return result;
}

__host__ __device__ __inline__ float3 phase(complex3 c)
{
	float3 result;
	result.x = phase(c.x);
	result.y = phase(c.y);
	result.z = phase(c.z);
	return result;
}

__host__ __device__ __inline__ complex3 pow(complex3 c, int exp)
{
	complex3 result = { pow(c.x, exp), pow(c.y, exp), pow(c.z, exp) };
	return result;
}

__host__ __device__ __inline__ complex3 sqrt(complex3 c)
{
	complex3 result = { sqrt(c.x), sqrt(c.y), sqrt(c.z) };
	return result;
}

#endif // COMPLEX_H