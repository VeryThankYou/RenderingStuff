#ifndef QUATERNION_H
#define QUATERNION_H

#include <cmath>
#include <sutil/vec_math.h>

class quaternion
{
public:

  /// Vector part of quaternion
  float3 qv;

  /// Scalar part of quaternion
  float qw;

  /// Construct undefined quaternion
  quaternion() {}

  /// Construct quaternion from vector and scalar
  quaternion(const float3& imaginary, float real = 1.0f) : qv(imaginary), qw(real) {}

  /// Construct quaternion from four scalars
  quaternion(float x, float y, float z, float w) : qv(make_float3(x, y, z)), qw(w) {}

  /// Construct quaternion from vector
  explicit quaternion(const float4& v) : qv(make_float3(v.x, v.y, v.z)), qw(v.w) {}

  /// Obtain angle of rotation and axis
  void get_rot(float& angle, float3& v)
  {
    angle = 2.0f*std::acos(qw);

    if(angle < 1.0e-7f)
      v = make_float3(1.0f, 0.0f, 0.0f);
    else
      v = qv*(1.0f/std::sin(angle));

    if(angle > M_PIf)
      v = -v;

    v = ::normalize(v);
  }

  /// Construct a Quaternion from an angle and axis of rotation.
  void make_rot(float angle, const float3& v)
  {
    angle /= 2.0f;
    qv = ::normalize(v)*sinf(angle);
    qw = cosf(angle);
  }

  /// Construct a Quaternion rotating from the direction given by the first argument to the direction given by the second.
  void make_rot(const float3& s, const float3& t)
  {
    float tmp = std::sqrtf(2.0f*(1.0f + dot(s, t)));
    qv = cross(s, t)*(1.0f/tmp);
    qw = tmp*0.5f;
  }
/*
  /// Get a 3x3 rotation matrix from a quaternion
  Matrix3x3 get_matrix3x3() const
  {
    float s = 2.0f/norm();
    float3 qv_qv = qv*qv;
    float3 qv_qw = qv*qw;
    Matrix3x3 result;
    // note that the all q_*q_ are used twice (optimize)
    result.setRow(0, make_float3(1.0f - s*(qv_qv.y + qv_qv.z), s*(qv.x*qv.y - qv_qw.z), s*(qv.x*qv.z + qv_qw.y)));
    result.setRow(1, make_float3(s*(qv.x*qv.y + qv_qw.z), 1.0f - s*(qv_qv.x + qv_qv.z), s*(qv.y*qv.z - qv_qw.x)));
    result.setRow(2, make_float3(s*(qv.x*qv.z - qv_qw.y), s*(qv.y*qv.z + qv_qw.x), 1.0f - s*(qv_qv.x + qv_qv.y)));
    return result;
  }

  /// Get a 4x4 rotation matrix from a quaternion
  Matrix4x4 get_matrix4x4() const
  {
    float s = 2.0f/norm();
    float3 qv_qv = qv*qv;
    float3 qv_qw = qv*qw;
    Matrix4x4 result;
    // note that the all q_*q_ are used twice (optimize)
    result.setRow(0, make_float4(1.0f - s*(qv_qv.y + qv_qv.z), s*(qv.x*qv.y - qv_qw.z), s*(qv.x*qv.z + qv_qw.y), 0.0f));
    result.setRow(1, make_float4(s*(qv.x*qv.y + qv_qw.z), 1.0f - s*(qv_qv.x + qv_qv.z), s*(qv.y*qv.z - qv_qw.x), 0.0f));
    result.setRow(2, make_float4(s*(qv.x*qv.z - qv_qw.y), s*(qv.y*qv.z + qv_qw.x), 1.0f - s*(qv_qv.x + qv_qv.y), 0.0f));
    result.setRow(3, make_float4(0.0f, 0.0f, 0.0f, 1.0f));
    return result;
  }

  /// Construct a Quaternion from a rotation matrix.
  void make_rot(const Matrix3x3& m)
  {
    float trace = m.getRow(0).x + m.getRow(1).y + m.getRow(2).z  + 1.0f;

    //If the trace of the matrix is greater than zero, then
    //perform an "instant" calculation.
    if(trace > 1.0e-7f)
    {
      float S = std::sqrt(trace)*2.0f;
      qv = make_float3(m.getRow(2).y - m.getRow(1).z, m.getRow(0).z - m.getRow(2).x, m.getRow(1).x - m.getRow(0).y);
      qv /= S;
      qw = 0.25f*S;
    }
    else
    {
      //If the trace of the matrix is equal to zero (or negative...) then identify
      //which major diagonal element has the greatest value.
      //Depending on this, calculate the following:

      if(m.getRow(0).x > m.getRow(1).y && m.getRow(0).x > m.getRow(2).z)	// Column 0: 
      {
        float S = std::sqrtf(1.0f + m.getRow(0).x - m.getRow(1).y - m.getRow(2).z)*2.0f;
        qv.x = 0.25f*S;
        qv.y = (m.getRow(1).x + m.getRow(0).y)/S;
        qv.z = (m.getRow(0).z + m.getRow(2).x)/S;
        qw = (m.getRow(2).y - m.getRow(1).z)/S;
      }
      else if(m.getRow(1).y > m.getRow(2).z)			// Column 1: 
      {
        float S = std::sqrtf(1.0f + m.getRow(1).y - m.getRow(0).x - m.getRow(2).z)*2.0f;
        qv.x = (m.getRow(1).x + m.getRow(0).y)/S;
        qv.y = 0.25f*S;
        qv.z = (m.getRow(2).y + m.getRow(1).z)/S;
        qw = (m.getRow(0).z - m.getRow(2).x)/S;
      }
      else                            // Column 2:
      {
        float S = std::sqrtf(1.0f + m.getRow(2).z - m.getRow(0).x - m.getRow(1).y)*2.0f;
        qv.x = (m.getRow(0).z + m.getRow(2).x)/S;
        qv.y = (m.getRow(2).y + m.getRow(1).z)/S;
        qv.z = 0.25f*S;
        qw = (m.getRow(1).x - m.getRow(0).y)/S;
      }
    }
    //The quaternion is then defined as:
    //  Q = | X Y Z W |
  }
*/
  //----------------------------------------------------------------------
  // Binary operators
  //----------------------------------------------------------------------

  /// Multiply two quaternions. (Combine their rotation)
  quaternion operator*(const quaternion& q) const
  {
    return quaternion(cross(qv, q.qv) + qv*q.qw + q.qv*qw, qw*q.qw - dot(qv, q.qv));
  }

  /// Multiply scalar onto quaternion.
  quaternion operator*(float scalar) const
  {
    return quaternion(qv*scalar, qw*scalar);
  }

  /// Add two quaternions.
  quaternion operator+(const quaternion& q) const
  {
    return quaternion(qv + q.qv, qw + q.qw);
  }

  /// Check equality of two quaternions
  bool operator==(const quaternion& q) const
  {
    return qv.x == q.qv.x && qv.y == q.qv.y && qv.z == q.qv.z && qw == q.qw;
  }

  //----------------------------------------------------------------------
  // Unary operators
  //----------------------------------------------------------------------

  /// Compute the additive inverse of the quaternion
  quaternion operator-() const { return quaternion(-qv, -qw); }

  /// Compute norm of quaternion
  float norm() const { return dot(qv, qv) + qw*qw; }

  /// Return conjugate quaternion
  quaternion conjugate() const { return quaternion(-qv, qw); }

  /// Compute the multiplicative inverse of the quaternion
  quaternion inverse() const { return quaternion(conjugate()*(1.0f/norm())); }

  /// Normalize quaternion.
  quaternion normalize() { return quaternion((*this)*(1.0f/norm())); }

  //----------------------------------------------------------------------
  // Application
  //----------------------------------------------------------------------

  /// Rotate vector according to quaternion
  float3 apply(const float3& vec) const
  {
    return ((*this)*quaternion(vec)*inverse()).qv;
  }

  /// Rotate vector according to unit quaternion
  float3 apply_unit(const float3& vec) const
  {
    return ((*this)*quaternion(vec)*conjugate()).qv;
  }
};

/// Create an identity quaternion
inline quaternion identity_quaternion()
{
  return quaternion(make_float3(0.0f));
}

inline quaternion operator*(float scalar, const quaternion& q)
{
  return q*scalar;
}

/** Perform linear interpolation of two quaternions.
The last argument is the parameter used to interpolate
between the two first. SLERP - invented by Shoemake -
is a good way to interpolate because the interpolation
is performed on the unit sphere.
*/
inline quaternion slerp(const quaternion& q0, const quaternion& q1, float t)
{
  float angle = std::acos(dot(q0.qv, q1.qv) + q0.qw*q1.qw);
  return (q0*std::sin((1.0f - t)*angle) + q1*std::sin(t*angle))*(1.0f/std::sin(angle));
}

/// Print quaternion to stream.
inline std::ostream& operator<<(std::ostream&os, const quaternion& v)
{
  os << "[ ";
  for(unsigned int i = 0; i<3; i++) os << *(&v.qv + i) << " ";
  os << "~ " << v.qw << " ";
  os << "]";

  return os;
}
#endif // QUATERNION_H
