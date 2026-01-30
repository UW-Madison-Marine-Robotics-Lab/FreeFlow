// SPDX-License-Identifier: MIT
// Copyright (c) 2025 Changyu Hu
//
// Commons Clause addition:
// This software is provided for non-commercial use only. See LICENSE file for details.

// --------------------------------------------------------------------------------
// Modifications Copyright 2026 Jiayi Jin
//
// This file has been significantly modified from its original version in
// the FreeFlow library. The original license and copyright
// notices are retained above.
//
// The modifications are provided under the terms of the license of this project.
// --------------------------------------------------------------------------------

#pragma once

#include "FSI_Simulator/common/CudaCommon.cuh"
#include "FSI_Simulator/common/Types.hpp"
#include <vector>
#include <string>

namespace fsi
{
    namespace control
    {

        void farthest_point_sampling(
            int vnum, const std::vector<vec3_t> &vpos, const std::vector<unsigned int> &tetvIdx,
            int cnum, std::vector<int> &v_ctrl, std::vector<real> &lbs_dist,
            std::string lbs_distance_type,
            bool random_first);

        void farthest_point_sampling(
            int vnum, const std::vector<vec2_t> &vpos, const std::vector<unsigned int> &trivIdx,
            int cnum, std::vector<int> &v_ctrl, std::vector<real> &lbs_dist,
            std::string lbs_distance_type,
            bool random_first);

        // user-defined control points setup
        void set_control_points(
            int vnum, const std::vector<vec3_t> &vpos, const std::vector<unsigned int> &tetvIdx,
            std::vector<int> &v_ctrl, std::vector<real> &lbs_dist,
            std::string lbs_distance_type
        );

        void set_control_points(
            int vnum, const std::vector<vec2_t> &vpos, const std::vector<unsigned int> &trivIdx,
            std::vector<int> &v_ctrl, std::vector<real> &lbs_dist,
            std::string lbs_distance_type
        );
    }
}