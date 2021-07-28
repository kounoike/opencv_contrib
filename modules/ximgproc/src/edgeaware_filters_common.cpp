/*
 *  By downloading, copying, installing or using the software you agree to this license.
 *  If you do not agree to this license, do not download, install,
 *  copy or use the software.
 *
 *
 *  License Agreement
 *  For Open Source Computer Vision Library
 *  (3 - clause BSD License)
 *
 *  Redistribution and use in source and binary forms, with or without modification,
 *  are permitted provided that the following conditions are met :
 *
 *  * Redistributions of source code must retain the above copyright notice,
 *  this list of conditions and the following disclaimer.
 *
 *  * Redistributions in binary form must reproduce the above copyright notice,
 *  this list of conditions and the following disclaimer in the documentation
 *  and / or other materials provided with the distribution.
 *
 *  * Neither the names of the copyright holders nor the names of the contributors
 *  may be used to endorse or promote products derived from this software
 *  without specific prior written permission.
 *
 *  This software is provided by the copyright holders and contributors "as is" and
 *  any express or implied warranties, including, but not limited to, the implied
 *  warranties of merchantability and fitness for a particular purpose are disclaimed.
 *  In no event shall copyright holders or contributors be liable for any direct,
 *  indirect, incidental, special, exemplary, or consequential damages
 *  (including, but not limited to, procurement of substitute goods or services;
 *  loss of use, data, or profits; or business interruption) however caused
 *  and on any theory of liability, whether in contract, strict liability,
 *  or tort(including negligence or otherwise) arising in any way out of
 *  the use of this software, even if advised of the possibility of such damage.
 */

#include "precomp.hpp"
#include "edgeaware_filters_common.hpp"
#include "dtfilter_cpu.hpp"

#include <opencv2/core/cvdef.h>
#include <opencv2/core/utility.hpp>
#include <opencv2/core/hal/intrin.hpp>
#include <cmath>
using namespace std;

#ifndef SQR
#define SQR(x) ((x)*(x))
#endif

#if CV_SSE
namespace
{

inline bool CPU_SUPPORT_SSE1()
{
    static const bool is_supported = cv::checkHardwareSupport(CV_CPU_SSE);
    return is_supported;
}

}  // end
#endif

namespace cv
{
namespace ximgproc
{

Ptr<DTFilter> createDTFilterRF(InputArray adistHor, InputArray adistVert, double sigmaSpatial, double sigmaColor, int numIters)
{
    return Ptr<DTFilter>(DTFilterCPU::createRF(adistHor, adistVert, sigmaSpatial, sigmaColor, numIters));
}

int getTotalNumberOfChannels(InputArrayOfArrays src)
{
    CV_Assert(src.isMat() || src.isUMat() || src.isMatVector() || src.isUMatVector());

    if (src.isMat() || src.isUMat())
    {
        return src.channels();
    }
    else if (src.isMatVector())
    {
        int cnNum = 0;
        const vector<Mat>& srcv = *static_cast<const vector<Mat>*>(src.getObj());
        for (unsigned i = 0; i < srcv.size(); i++)
            cnNum += srcv[i].channels();
        return cnNum;
    }
    else if (src.isUMatVector())
    {
        int cnNum = 0;
        const vector<UMat>& srcv = *static_cast<const vector<UMat>*>(src.getObj());
        for (unsigned i = 0; i < srcv.size(); i++)
            cnNum += srcv[i].channels();
        return cnNum;
    }
    else
    {
        return 0;
    }
}

void checkSameSizeAndDepth(InputArrayOfArrays src, Size &sz, int &depth)
{
    CV_Assert(src.isMat() || src.isUMat() || src.isMatVector() || src.isUMatVector());

    if (src.isMat() || src.isUMat())
    {
        CV_Assert(!src.empty());
        sz = src.size();
        depth = src.depth();
    }
    else if (src.isMatVector())
    {
        const vector<Mat>& srcv = *static_cast<const vector<Mat>*>(src.getObj());
        CV_Assert(srcv.size() > 0);
        for (unsigned i = 0; i < srcv.size(); i++)
        {
            CV_Assert(srcv[i].depth() == srcv[0].depth());
            CV_Assert(srcv[i].size() == srcv[0].size());
        }
        sz = srcv[0].size();
        depth = srcv[0].depth();
    }
    else if (src.isUMatVector())
    {
        const vector<UMat>& srcv = *static_cast<const vector<UMat>*>(src.getObj());
        CV_Assert(srcv.size() > 0);
        for (unsigned i = 0; i < srcv.size(); i++)
        {
            CV_Assert(srcv[i].depth() == srcv[0].depth());
            CV_Assert(srcv[i].size() == srcv[0].size());
        }
        sz = srcv[0].size();
        depth = srcv[0].depth();
    }
}

namespace intrinsics
{

inline float getFloatSignBit()
{
    union
    {
        int signInt;
        float signFloat;
    };
    signInt = 0x80000000;

    return signFloat;
}

void add_(float *dst, float *src1, int w)
{
    int j = 0;
#if CV_SIMD128
    v_float32x4 a, b;
    for (; j < w - 3; j += 4)
    {
        a = v_load(src1 + j);
        b = v_load(dst + j);
        b += a;
        v_store(dst + j, b);
    }
#endif
    for (; j < w; j++)
        dst[j] += src1[j];
}

void mul(float *dst, float *src1, float *src2, int w)
{
    int j = 0;
#if CV_SIMD128
    v_float32x4 a, b;
    for (; j < w - 3; j += 4)
    {
        a = v_load(src1 + j);
        b = v_load(src2 + j);
        b *= a;
        v_store(dst + j, b);
    }
#endif
    for (; j < w; j++)
        dst[j] = src1[j] * src2[j];
}

void mul(float *dst, float *src1, float src2, int w)
{
    int j = 0;
#if CV_SIMD128
    float f[4] = {src2, src2, src2, src2};
    v_float32x4 a, b;
    b = v_load(f);
    for (; j < w - 3; j += 4)
    {
        a = v_load(src1 + j);
        a *= b;
        v_store(dst + j, a);
    }
#endif
    for (; j < w; j++)
        dst[j] = src1[j]*src2;
}

void mad(float *dst, float *src1, float alpha, float beta, int w)
{
    int j = 0;
#if CV_SIMD128
    v_float32x4 a, b, c, d;
    float fa[4] = {alpha, alpha, alpha, alpha};
    float fb[4] = {beta, beta, beta, beta};
    a = v_load(fa);
    b = v_load(fb);
    for (; j < w - 3; j += 4)
    {
        c = v_load(src1 + j);
        d = c * a + b;
        v_store(dst + j, d);
    }
#endif
    for (; j < w; j++)
        dst[j] = alpha*src1[j] + beta;
}

void sqr_(float *dst, float *src1, int w)
{
    int j = 0;
#if CV_SIMD128
    v_float32x4 a;
    for (; j < w - 3; j += 4)
    {
        a = v_load(src1 + j);
        a *= a;
        v_store(dst + j, a);
    }
#endif
    for (; j < w; j++)
        dst[j] = src1[j] * src1[j];
}

void sqr_dif(float *dst, float *src1, float *src2, int w)
{
    int j = 0;
#if CV_SIMD128
    v_float32x4 a, b;
    for (; j < w - 3; j += 4)
    {
        a = v_load(src1 + j);
        b = v_load(src2 + j);
        a -= b;
        a *= a;
        v_store(dst + j, a);
    }
#endif
    for (; j < w; j++)
        dst[j] = (src1[j] - src2[j])*(src1[j] - src2[j]);
}

void add_mul(float *dst, float *src1, float *src2, int w)
{
    int j = 0;
#if CV_SIMD128
    v_float32x4 a, b, c;
    for (; j < w - 3; j += 4)
    {
        a = v_load(src1 + j);
        b = v_load(src2 + j);
        c = v_load(dst + j);
        c += a * b;
        v_store(dst + j, c);
    }
#endif
    for (; j < w; j++)
    {
        dst[j] += src1[j] * src2[j];
    }
}

void add_sqr(float *dst, float *src1, int w)
{
    int j = 0;
#if CV_SIMD
    v_float32x4 a, b;
    for (; j < w - 3; j += 4)
    {
        a = v_load(src1 + j);
        b = v_load(dst + j);
        b += a * a;
        v_store(dst + j, b);
    }
#endif
    for (; j < w; j++)
    {
        dst[j] += src1[j] * src1[j];
    }
}

void add_sqr_dif(float *dst, float *src1, float *src2, int w)
{
    int j = 0;
#if CV_SIMD128
    v_float32x4 a, b, c;
    for (; j < w - 3; j += 4)
    {
        a = v_load(src1 + j);
        b = v_load(src2 + j);
        c = v_load(dst + j);
        a -= b;
        a *= a;
        c += a;
        v_store(dst + j, c);
    }
#endif
    for (; j < w; j++)
    {
        dst[j] += (src1[j] - src2[j])*(src1[j] - src2[j]);
    }
}

void sub_mul(float *dst, float *src1, float *src2, int w)
{
    int j = 0;
#if CV_SIMD128
    v_float32x4 a, b, c;
    for (; j < w - 3; j += 4)
    {
        a = v_load(src1 + j);
        b = v_load(src2 + j);
        c = v_load(dst + j);
        c -= a * b;
        v_store(dst + j, c);
    }
#endif
    for (; j < w; j++)
        dst[j] -= src1[j] * src2[j];
}

void sub_mad(float *dst, float *src1, float *src2, float c0, int w)
{
    int j = 0;
#if CV_SIMD128
    float fc0[4] = {c0, c0, c0, c0};
    v_float32x4 a, b, c;
    v_float32x4 cnst = v_load(fc0);
    for (; j < w - 3; j += 4)
    {
        a = v_load(src1 + j);
        b = v_load(src2 + j);
        c = v_load(dst + j);
        c -= a * b - cnst;
        v_store(dst + j, c);
    }
#endif
    for (; j < w; j++)
        dst[j] -= src1[j] * src2[j] + c0;
}

void det_2x2(float *dst, float *a00, float *a01, float *a10, float *a11, int w)
{
    int j = 0;
#if CV_SIMD128
    v_float32x4 v_a00, v_a01, v_a10, v_a11, d;
    for (; j < w - 3; j += 4)
    {
        v_a00 = v_load(a00 + j);
        v_a01 = v_load(a01 + j);
        v_a10 = v_load(a10 + j);
        v_a11 = v_load(a11 + j);
        d = v_a00 * v_a11 - v_a01 * v_a10;
        v_store(dst + j, d);
    }
#endif
    for (; j < w; j++)
        dst[j] = a00[j]*a11[j] - a01[j]*a10[j];
}

void div_det_2x2(float *a00, float *a01, float *a11, int w)
{
    int j = 0;
#if CV_SIMD128
    // float sign_mask[4] = {getFloatSignBit(), getFloatSignBit(), getFloatSignBit(), getFloatSignBit()};

    v_float32x4 _a00, _a01, _a11, _det;
    for (; j < w - 3; j += 4)
    {
        _a00 = v_load(a00 + j);
        _a01 = v_load(a01 + j);
        _a11 = v_load(a11 + j);
        _det = _a00 * _a11 - _a01 * _a01;
        _a00 /= _det;
        _a01 /= _det;
        _a11 /= _det;
        v_store(a01 + j, _a01);
        v_store(a00 + j, _a00);
        v_store(a11 + j, _a11);
    }
#endif
    for (; j < w; j++)
    {
        float det = a00[j] * a11[j] - a01[j] * a01[j];
        a00[j] /= det;
        a11[j] /= det;
        a01[j] /= -det;
    }
}

void div_1x(float *a1, float *b1, int w)
{
    int j = 0;
#if CV_SIMD128
    v_float32x4 _a1, _b1;
    for (; j < w - 3; j += 4)
    {
        _b1 = v_load(b1 + j);
        _a1 = v_load(a1 + j);
        _a1 /= _b1;
        v_store(a1 + j, _a1);
    }
#endif
    for (; j < w; j++)
    {
        a1[j] /= b1[j];
    }
}

void inv_self(float *src, int w)
{
    int j = 0;
#if CV_SIMD128
    float f1[4] = {1.0f, 1.0f, 1.0f, 1.0f};
    v_float32x4 _1 = v_load(f1);
    v_float32x4 a;
    for (; j < w - 3; j += 4)
    {
        a = v_load(src + j);
        v_store(src + j, _1 / a);
    }
#endif
    for (; j < w; j++)
    {
        src[j] = 1.0f / src[j];
    }
}

void sqrt_(float *dst, float *src, int w)
{
    int j = 0;
#if CV_SIMD128
    v_float32x4 a;
    for (; j < w - 3; j += 4)
    {
        a = v_sqrt(v_load(src + j));
        v_store(dst + j, a);
    }
#endif
    for (; j < w; j++)
        dst[j] = sqrt(src[j]);
}

void min_(float *dst, float *src1, float *src2, int w)
{
    int j = 0;
#if CV_SIMD128
    v_float32 a, b;
    for (; j < w - 3; j += 4)
    {
        a = v_load(src1 + j);
        b = v_load(src2 + j);
        b = v_min(b, a);

        v_store(dst + j, b);
    }
#endif
    for (; j < w; j++)
        dst[j] = std::min(src1[j], src2[j]);
}

void rf_vert_row_pass(float *curRow, float *prevRow, float alphaVal, int w)
{
    int j = 0;
#if CV_SIMD128
    v_float32x4 cur, prev, res;
    float falpha[4] = {alphaVal, alphaVal, alphaVal, alphaVal};
    v_float32x4 alpha = v_load(falpha);
    for (; j < w - 3; j += 4)
    {
        cur = v_load(curRow + j);
        prev = v_load(prevRow + j);
        cur += alpha * (prev - cur);
        v_store(curRow + j, cur);
    }
#endif
    for (; j < w; j++)
        curRow[j] += alphaVal*(prevRow[j] - curRow[j]);
}

} //end of cv::ximgproc::intrinsics

} //end of cv::ximgproc
} //end of cv
