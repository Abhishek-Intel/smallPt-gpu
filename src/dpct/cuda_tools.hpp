#pragma once

//-----------------------------------------------------------------------------
// CUDA Includes
//-----------------------------------------------------------------------------
#pragma region

#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>


#pragma endregion

//-----------------------------------------------------------------------------
// System Includes
//-----------------------------------------------------------------------------
#pragma region

#include <cstdio>
#include <cstdlib>

#pragma endregion

//-----------------------------------------------------------------------------
// Declarations and Definitions
//-----------------------------------------------------------------------------
namespace smallpt {

	inline void HandleError(int err, const char* file, int line) {
		
	}
}

//-----------------------------------------------------------------------------
// Defines
//-----------------------------------------------------------------------------
#pragma region

#define HANDLE_ERROR(err) (HandleError( err, __FILE__, __LINE__ ))


#define HANDLE_NULL(a) {if (a == NULL) { \
	std::printf( "Host memory failed in %s at line %d\n", __FILE__, __LINE__ ); \
    std::exit( EXIT_FAILURE );}}

#pragma endregion