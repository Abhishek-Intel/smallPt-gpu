#pragma once

//-----------------------------------------------------------------------------
// Includes
//-----------------------------------------------------------------------------
#pragma region

#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include "math.hpp"
#include <cmath>


#pragma endregion

//-----------------------------------------------------------------------------
// Declarations and Definitions
//-----------------------------------------------------------------------------
namespace smallpt {

	//-------------------------------------------------------------------------
	// Vector3
	//-------------------------------------------------------------------------

	struct Vector3 {

	public:

		//---------------------------------------------------------------------
		// Constructors and Destructors
		//---------------------------------------------------------------------

		  explicit Vector3(double xyz = 0.0) noexcept
			: Vector3(xyz, xyz, xyz) {}
		  Vector3(double x, double y, double z) noexcept
			: m_x(x), m_y(y), m_z(z) {}
		Vector3(const Vector3& v) noexcept = default;
		Vector3(Vector3&& v) noexcept = default;
		~Vector3() = default;

		//---------------------------------------------------------------------
		// Assignment Operators
		//---------------------------------------------------------------------

		Vector3& operator=(const Vector3& v) = default;
		Vector3& operator=(Vector3&& v) = default;

		//---------------------------------------------------------------------
		// Member Methods
		//---------------------------------------------------------------------

		 bool HasNaNs() const {
			return std::isnan(m_x) || std::isnan(m_y) || std::isnan(m_z);
		}

		 const Vector3 operator-() const {
			return { -m_x, -m_y, -m_z };
		}

		 const Vector3 operator+(const Vector3& v) const {
			return { m_x + v.m_x, m_y + v.m_y, m_z + v.m_z };
		}
		 const Vector3 operator-(const Vector3& v) const {
			return { m_x - v.m_x, m_y - v.m_y, m_z - v.m_z };
		}
		 const Vector3 operator*(const Vector3& v) const {
			return { m_x * v.m_x, m_y * v.m_y, m_z * v.m_z };
		}
		 const Vector3 operator/(const Vector3& v) const {
			return { m_x / v.m_x, m_y / v.m_y, m_z / v.m_z };
		}
		 const Vector3 operator+(double a) const {
			return { m_x + a, m_y + a, m_z + a };
		}
		 const Vector3 operator-(double a) const {
			return { m_x - a, m_y - a, m_z - a };
		}
		 const Vector3 operator*(double a) const {
			return { m_x * a, m_y * a, m_z * a };
		}
		 const Vector3 operator/(double a) const {
			const double inv_a = 1.0 / a;
			return { m_x * inv_a, m_y * inv_a, m_z * inv_a };
		}

		 Vector3& operator+=(const Vector3& v) {
			m_x += v.m_x;
			m_y += v.m_y;
			m_z += v.m_z;
			return *this;
		}
		 Vector3& operator-=(const Vector3& v) {
			m_x -= v.m_x;
			m_y -= v.m_y;
			m_z -= v.m_z;
			return *this;
		}
		 Vector3& operator*=(const Vector3& v) {
			m_x *= v.m_x;
			m_y *= v.m_y;
			m_z *= v.m_z;
			return *this;
		}
		 Vector3& operator/=(const Vector3& v) {
			m_x /= v.m_x;
			m_y /= v.m_y;
			m_z /= v.m_z;
			return *this;
		}
		 Vector3& operator+=(double a) {
			m_x += a;
			m_y += a;
			m_z += a;
			return *this;
		}
		 Vector3& operator-=(double a) {
			m_x -= a;
			m_y -= a;
			m_z -= a;
			return *this;
		}
		 Vector3& operator*=(double a) {
			m_x *= a;
			m_y *= a;
			m_z *= a;
			return *this;
		}
		 Vector3& operator/=(double a) {
			const double inv_a = 1.0 / a;
			m_x *= inv_a;
			m_y *= inv_a;
			m_z *= inv_a;
			return *this;
		}

		 double Dot(const Vector3& v) const {
			return m_x * v.m_x + m_y * v.m_y + m_z * v.m_z;
		}
		 const Vector3 Cross(const Vector3& v) const {
			return {
				m_y * v.m_z - m_z * v.m_y,
				m_z * v.m_x - m_x * v.m_z,
				m_x * v.m_y - m_y * v.m_x
			};
		}

		 bool operator==(const Vector3& rhs) const {
			return m_x == rhs.m_x && m_y == rhs.m_y && m_z == rhs.m_z;
		}
		 bool operator!=(const Vector3& rhs) const {
			return !(*this == rhs);
		}

		 double& operator[](std::size_t i) {
			return (&m_x)[i];
		}
		 double operator[](std::size_t i) const {
			return (&m_x)[i];
		}

		 std::size_t MinDimension() const {
			return (m_x < m_y && m_x < m_z) ? 0u : ((m_y < m_z) ? 1u : 2u);
		}
		 std::size_t MaxDimension() const {
			return (m_x > m_y && m_x > m_z) ? 0u : ((m_y > m_z) ? 1u : 2u);
		}
		 double Min() const {
			return cl::sycl::fmin(m_x, cl::sycl::fmin(m_y, m_z));
		}
		 double Max() const {
			return cl::sycl::fmax(m_x, cl::sycl::fmax(m_y, m_z));
		}

		 double Norm2_squared() const {
			return m_x * m_x + m_y * m_y + m_z * m_z;
		}

		 double Norm2() const {
			return std::sqrt(Norm2_squared());
		}

		 void Normalize() {
			const double a = 1.0 / Norm2();
			m_x *= a;
			m_y *= a;
			m_z *= a;
		}

		//---------------------------------------------------------------------
		// Member Variables
		//---------------------------------------------------------------------

		double m_x, m_y, m_z;
	};

	//-------------------------------------------------------------------------
	// Vector3 Utilities
	//-------------------------------------------------------------------------

	 inline const Vector3 operator+(double a, const Vector3& v) {
		return { a + v.m_x, a + v.m_y, a + v.m_z };
	}

	 inline const Vector3 operator-(double a, const Vector3& v) {
		return { a - v.m_x, a - v.m_y, a - v.m_z };
	}

	 inline const Vector3 operator*(double a, const Vector3& v) {
		return { a * v.m_x, a * v.m_y, a * v.m_z };
	}

	 inline const Vector3 operator/(double a, const Vector3& v) {
		return { a / v.m_x, a / v.m_y, a / v.m_z };
	}

	 inline const Vector3 Sqrt(const Vector3& v) {
		return {
			std::sqrt(v.m_x),
			std::sqrt(v.m_y),
			std::sqrt(v.m_z)
		};
	}

	 inline const Vector3 Pow(const Vector3& v, double a) {
		return {
			std::pow(v.m_x, a),
			std::pow(v.m_y, a),
			std::pow(v.m_z, a)
		};
	}

	 inline const Vector3 Abs(const Vector3& v) {
		return {
			std::fabs(v.m_x),
			std::fabs(v.m_y),
			std::fabs(v.m_z)
		};
	}

	 inline const Vector3 Min(const Vector3& v1, const Vector3& v2) {
		return {
			cl::sycl::fmin(v1.m_x, v2.m_x),
			cl::sycl::fmin(v1.m_y, v2.m_y),
			cl::sycl::fmin(v1.m_z, v2.m_z)
		};
	}

	 inline const Vector3 Max(const Vector3& v1, const Vector3& v2) {
		return {
			cl::sycl::fmax(v1.m_x, v2.m_x),
			cl::sycl::fmax(v1.m_y, v2.m_y),
			cl::sycl::fmax(v1.m_z, v2.m_z)
		};
	}

	 inline const Vector3 Round(const Vector3& v) {
		return {
			std::round(v.m_x),
			std::round(v.m_y),
			std::round(v.m_z)
		};
	}

	 inline const Vector3 Floor(const Vector3& v) {
		return {
			std::floor(v.m_x),
			std::floor(v.m_y),
			std::floor(v.m_z)
		};
	}

	 inline const Vector3 Ceil(const Vector3& v) {
		return {
			std::ceil(v.m_x),
			std::ceil(v.m_y),
			std::ceil(v.m_z)
		};
	}

	 inline const Vector3 Trunc(const Vector3& v) {
		return {
			std::trunc(v.m_x),
			std::trunc(v.m_y),
			std::trunc(v.m_z)
		};
	}

	 inline const Vector3 Clamp(const Vector3& v, 
										  double low = 0.0, 
										  double high = 1.0) {
		return {
			Clamp(v.m_x, low, high),
			Clamp(v.m_y, low, high),
			Clamp(v.m_z, low, high) }
		;
	}

	 inline const Vector3 Lerp(double a, 
										 const Vector3& v1, 
										 const Vector3& v2) {
		return v1 + a * (v2 - v1);
	}

	template< std::size_t X, std::size_t Y, std::size_t Z >
	 inline const Vector3 Permute(const Vector3& v) {
		return { v[X], v[Y], v[Z] };
	}

	 inline const Vector3 Normalize(const Vector3& v) {
		const double a = 1.0 / v.Norm2();
		return a * v;
	}
}