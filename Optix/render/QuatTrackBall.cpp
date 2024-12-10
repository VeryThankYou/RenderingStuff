/* ----------------------------------------------------------------------- *
 * This file is part of GEL, http://www.imm.dtu.dk/GEL
 * Copyright (C) the authors and DTU Informatics
 * For license and list of authors, see ../../doc/intro.pdf
 * ----------------------------------------------------------------------- */

#include <iostream>
#include <sutil/vec_math.h>
#include "quaternion.h"
#include "QuatTrackBall.h"

using namespace std;

QuatTrackBall::QuatTrackBall(const float3& _centre, 
			                       float _eye_dist,
			                       unsigned _width,
                             unsigned _height) :
centre(_centre), width(_width), height(_height), eye_dist(_eye_dist)
{
  // This size should really be based on the distance from the center of
  // rotation to the point on the object underneath the mouse.  That
  // point would then track the mouse as closely as possible.  This is a
  // simple example, though, so that is left as an exercise.
  ballsize = 1.7f;
  screen_centre = make_uint2(width/2, height/2);
  qrot = quaternion(0.0, 0.0, 0.0, 1.0);
  qinc = quaternion(0.0, 0.0, 0.0, 1.0);
  trans = make_float2(0.0, 0.0);
}
    
void QuatTrackBall::grab_ball(TrackBallAction act, const float2& v)
{
  if(v.x < 0 || v.x >= width || v.y < 0 || v.y >= height)
    return;      

  set_position(scalePoint(v));
  current_action = act;
}
    
void QuatTrackBall::roll_ball(const float2& v)
{
  if(v.x < 0 || v.x >= width || v.y < 0 || v.y >= height)
    return;      

	float2 w = scalePoint(v); 
	
	switch (current_action) 
  {
	case ORBIT_ACTION:
    orbit(w);
    break;                
	case PAN_ACTION:
    pan(w);
    break;                
	case DOLLY_ACTION:
    dolly(w);
    break;
  case NO_ACTION:
  default:
    break;
  }
	last_pos = w;
}
    
// Call this when the user does a mouse down.  
// Stop the trackball glide, then remember the mouse
// down point (for a future rotate, pan or zoom).
void QuatTrackBall::set_position(const float2& _last_pos) 
{
	stop_spin();
	last_pos = _last_pos;
}
    
// Rotationaly spin the trackball by the current increment.
// Use this to implement rotational glide.
void QuatTrackBall::do_spin() 
{
  qrot = qrot*qinc;
}
    
// Cease any rotational glide by zeroing the increment.
void QuatTrackBall::stop_spin() 
{
  qinc = quaternion(0.0, 0.0, 0.0, 1.0);
}
    
void QuatTrackBall::orbit(const float2& new_v) 
{
  calcRotation(new_v);
  do_spin();	
}
    
void QuatTrackBall::pan(const float2& new_v) 
{
  trans += (new_v - last_pos) * make_float2(eye_dist);
}
    
void QuatTrackBall::dolly(const float2& new_v) 
{
  eye_dist += (new_v.y - last_pos.y) * eye_dist;
  eye_dist = fmaxf(eye_dist, 0.01f);
}
    
void QuatTrackBall::calcRotation(const float2& new_pos) 
{
  // Check for zero rotation
  if (new_pos.x == last_pos.x && new_pos.y == last_pos.y) 
    qinc = quaternion(0.0f, 0.0f, 0.0f, 1.0f);
  else
  {
    // Form two vectors based on input points, find rotation axis
    float3 p1 = make_float3(new_pos.x, new_pos.y, projectToSphere(new_pos));
    float3 p2 = make_float3(last_pos.x, last_pos.y, projectToSphere(last_pos));
    qinc.make_rot(normalize(p1), normalize(p2));
    /*
    Vec3f q = cross(p1, p2);		// axis of rotation from p1 and p2 
    float L = sqrt(1.0f-dot(q,q) / (dot(p1,p1) * dot(p2,p2)));
		
    q.normalize();				// q' = axis of rotation 
    q *= sqrt((1.0f - L)/2.0f);	// q' = q' * sin(phi)
		
    qinc = quaternion(q[0],q[1],q[2],sqrt((1 + L)/2));
    */
  }
}
    
// Project an x,y pair onto a sphere of radius r OR a hyperbolic sheet
// if we are away from the center of the sphere.
float QuatTrackBall::projectToSphere(const float2& v) 
{
#ifndef M_SQRT_2
  const float M_SQRT_2 = 0.707106781187f;
#endif

  float d = length(v);
  float t = ballsize*M_SQRT_2;
  float z;
  
  // Inside sphere 
  if(d < ballsize) 
    z = sqrtf(ballsize*ballsize - d*d);
  else if(d < t)
    z = 0.0;
  // On hyperbola 
  else 
    z = t*t/d;

  return z;
}
    
// Scales integer point to the range [-1, 1]
float2 QuatTrackBall::scalePoint(const float2& v) const
{
  float2 w = make_float2(v.x, height - v.y);
  w -= make_float2(screen_centre);
  w.x /= static_cast<float>(width);
  w.y /= static_cast<float>(height);
  w = fminf(make_float2(1.0f), fmaxf(make_float2(-1.0f), 2.0f*w));
  return w; 
}
    
void QuatTrackBall::get_view_param(float3& eye, float3& _centre, float3& up) const
{
  up  = qrot.apply_unit(make_float3(0.0f, 1.0f, 0.0f));
  float3 right = qrot.apply_unit(make_float3(1.0f, 0.0f, 0.0f));
  _centre = centre - up * trans.y - right * trans.x; 
  eye = qrot.apply_unit(make_float3(0.0f, 0.0f, 1.0f)*eye_dist) + _centre;
}
    
bool QuatTrackBall::is_spinning() const
{
  if(dot(qinc.qv, qinc.qv) > 5.0e-6f)
    return true;
  return false;
}
