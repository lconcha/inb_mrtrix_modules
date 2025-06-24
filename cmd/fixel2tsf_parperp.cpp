/* Copyright (c) 2008-2024 the MRtrix3 contributors.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 *
 * Covered Software is provided under this License on an "as is"
 * basis, without warranty of any kind, either expressed, implied, or
 * statutory, including, without limitation, warranties that the
 * Covered Software is free of defects, merchantable, fit for a
 * particular purpose or non-infringing.
 * See the Mozilla Public License v. 2.0 for more details.
 *
 * For more details, see http://www.mrtrix.org/.
 */

#include "command.h"
#include "progressbar.h"
#include "algo/loop.h"

#include "image.h"
#include "fixel/helpers.h"
#include "fixel/keys.h"
#include "fixel/types.h"

#include "dwi/tractography/file.h"
#include "dwi/tractography/scalar_file.h"
#include "dwi/tractography/streamline.h"

#include "dwi/tractography/mapping/loader.h"
#include "dwi/tractography/mapping/mapper.h"


using namespace MR;
using namespace App;

using Fixel::index_type;

#define DEFAULT_ANGULAR_THRESHOLD 45.0

void usage ()
{
  AUTHOR = "David Raffelt (david.raffelt@florey.edu.au)";
  AUTHOR = "Modified by Luis Concha (lconcha@unam.mx)";

  SYNOPSIS = "Map fixel values to a track scalar file based on an input tractogram";

  DESCRIPTION
  + "This command is useful for visualising all brain fixels (e.g. the output from fixelcfestats) in 3D.";

  ARGUMENTS
  + Argument ("fixel_in", "the input fixel data file (within the fixel directory)").type_image_in ()
  + Argument ("tracks",   "the input track file ").type_tracks_in ()
  + Argument ("tsf",      "the output track scalar file").type_file_out ();

  OPTIONS
  + Option ("angle", "the max anglular threshold for computing correspondence "
                     "between a fixel direction and track tangent "
                     "(default = " + str(DEFAULT_ANGULAR_THRESHOLD, 2) + " degrees)")
  + Argument ("value").type_float (0.001, 90.0)
  + Option ("tsf_perp", "optional output track scalar file for the most perpendicular fixel values")
  + Argument ("tsf_perp").type_file_out ()
  + Option ("tsf_perpav", "output tsf file for average of all except most parallel fixel")
  + Argument ("tsf_perpav").type_file_out ();
}

using SetVoxelDir = DWI::Tractography::Mapping::SetVoxelDir;

void run ()
{
  auto in_data_image = Fixel::open_fixel_data_file<float> (argument[0]);
  if (in_data_image.size(2) != 1)
    throw Exception ("Only a single scalar value for each fixel can be output as a track scalar file, "
                     "therefore the input fixel data file must have dimension Nx1x1");

  Header in_index_header = Fixel::find_index_header (Fixel::get_fixel_directory (argument[0]));
  auto in_index_image = in_index_header.get_image<index_type>();
  auto in_directions_image = Fixel::find_directions_header (Fixel::get_fixel_directory (argument[0])).get_image<float>().with_direct_io();

  DWI::Tractography::Properties properties;
  DWI::Tractography::Reader<float> reader (argument[1], properties);
  properties.comments.push_back ("Created using fixel2tsf");
  properties.comments.push_back ("Source fixel image: " + Path::basename (argument[0]));
  properties.comments.push_back ("Source track file: " + Path::basename (argument[1]));

  DWI::Tractography::ScalarWriter<float> tsf_writer (argument[2], properties);

  // Optional perpendicular TSF writer
  std::unique_ptr<DWI::Tractography::ScalarWriter<float>> tsf_writer_perp;
  std::string tsf_perp_filename = get_option_value("tsf_perp", std::string());
  if (!tsf_perp_filename.empty()) {
    tsf_writer_perp.reset(new DWI::Tractography::ScalarWriter<float>(tsf_perp_filename, properties));
  }

  // Optional perpendicular_av TSF writer
  std::unique_ptr<DWI::Tractography::ScalarWriter<float>> tsf_writer_perpav;
  std::string tsf_perpav_filename = get_option_value("tsf_perpav", std::string());
  if (!tsf_perpav_filename.empty()) {
    tsf_writer_perpav.reset(new DWI::Tractography::ScalarWriter<float>(tsf_perpav_filename, properties));
  }

  
  float angular_threshold = get_option_value ("angle", DEFAULT_ANGULAR_THRESHOLD);
  const float angular_threshold_dp = cos (angular_threshold * (Math::pi / 180.0));

  const size_t num_tracks = properties["count"].empty() ? 0 : to<int> (properties["count"]);

  DWI::Tractography::Mapping::TrackMapperBase mapper (in_index_image);
  mapper.set_use_precise_mapping (true);

  ProgressBar progress ("mapping fixel values to streamline points", num_tracks);
  DWI::Tractography::Streamline<float> tck;
  DWI::Tractography::TrackScalar<float> scalars;

  const Transform transform (in_index_image);
  Eigen::Vector3d voxel_pos;

  while (reader (tck)) {
    SetVoxelDir dixels;
    mapper (tck, dixels);
    scalars.clear();
    scalars.set_index (tck.get_index());
    scalars.resize (tck.size(), 0.0f);

    DWI::Tractography::TrackScalar<float> scalars_perp, scalars_avg;
    if (tsf_writer_perp) {
      scalars_perp.clear();
      scalars_perp.set_index (tck.get_index());
      scalars_perp.resize (tck.size(), 0.0f);
    }
    if (tsf_writer_perpav) {
      scalars_avg.clear();
      scalars_avg.set_index (tck.get_index());
      scalars_avg.resize (tck.size(), 0.0f);
    }

    for (size_t p = 0; p < tck.size(); ++p) {
      std::cout << "Processing point " << p << " of " << tck.size()-1 << std::endl;
      voxel_pos = transform.scanner2voxel * tck[p].cast<default_type> ();
      std::cout << "  Voxel position: " << voxel_pos.transpose() << std::endl;
      for (SetVoxelDir::const_iterator d = dixels.begin(); d != dixels.end(); ++d) {
        if ((int)round(voxel_pos[0]) == (*d)[0] && (int)round(voxel_pos[1]) == (*d)[1] && (int)round(voxel_pos[2]) == (*d)[2]) {
          assign_pos_of (*d).to (in_index_image);
          Eigen::Vector3f dir = d->get_dir().cast<float>();
          dir.normalize();
          float largest_dp = 0.0f;
          int32_t closest_fixel_index = -1;
          float smallest_dp = 2.0f;
          int32_t perp_fixel_index = -1;

          in_index_image.index(3) = 0;
          index_type num_fixels_in_voxel = in_index_image.value();
          std::cout << "  Number of fixels in voxel: " << num_fixels_in_voxel << std::endl;
          in_index_image.index(3) = 1;
          index_type offset = in_index_image.value();

          for (size_t fixel = 0; fixel < num_fixels_in_voxel; ++fixel) {
            in_directions_image.index(0) = offset + fixel;
            const float dp = abs (dir.dot (Eigen::Vector3f (in_directions_image.row(1))));
            in_data_image.index(0) = offset + fixel;
            const float value = in_data_image.value();
            std::cout << "    Fixel index: " << fixel << ", dp: " << dp << ", scalar:" << value << std::endl;
            if (dp > largest_dp) {
              largest_dp = dp;
              closest_fixel_index = fixel;
            }
            if (dp < smallest_dp) {
              smallest_dp = dp;
              perp_fixel_index = fixel;
            }
          }

          std::cout << "  Largest dp: " << largest_dp << ", closest fixel index: " << closest_fixel_index
                    << ", most perpendicular fixel index: " << perp_fixel_index << std::endl;
          // Most parallel
          if (largest_dp > angular_threshold_dp) {
            in_data_image.index(0) = offset + closest_fixel_index;
            const float value = in_data_image.value();
            if (std::isfinite (value)) {
              std::cout << "  Parallel scalar value is: " << value  << std::endl;
              scalars[p] = value;
            } else {
              scalars[p] = NAN;
            }
          } else {
            std::cout << "  [WARN] Largest dp below threshold (" << angular_threshold_dp << "), setting parallel scalar to NAN." << std::endl;
            scalars[p] = NAN;
          }
          // Most perpendicular
          if (tsf_writer_perp) {
            if (num_fixels_in_voxel == 1) {
              // If there's only one fixel, we can directly assign it
              scalars_perp[p] = NaN;
              std::cout << "  [WARN] Only one fixel found, setting perp to NAN." << std::endl;
            } else {
                in_data_image.index(0) = offset + perp_fixel_index;
                const float value_perp = in_data_image.value();
                scalars_perp[p] = value_perp;
                std::cout << "  Perpendicular fixel value: " << value_perp << std::endl;
            }
          }
          // Average of all except most parallel
          if (tsf_writer_perpav) {
            if (num_fixels_in_voxel == 1) {
              // If there's only one fixel, we can directly assign it
              scalars_avg[p] = NaN;
              std::cout << "  [WARN] Only one fixel found, setting perpav to NAN." << std::endl;
            } else {
              float sum = 0.0f;
              int count = 0;
              for (size_t fixel = 0; fixel < num_fixels_in_voxel; ++fixel) {
                if ((int)fixel == closest_fixel_index) continue;
                in_data_image.index(0) = offset + fixel;
                float v = in_data_image.value();
                if (std::isfinite(v)) {
                  sum += v;
                  ++count;
                }
              }
              scalars_avg[p] = sum / count;    
              std::cout << "  Average of all except most parallel fixel: " << scalars_avg[p] << std::endl;          
            }
          }
          break;
        }
      }
    }
    tsf_writer (scalars);
    if (tsf_writer_perp) (*tsf_writer_perp)(scalars_perp);
    if (tsf_writer_perpav) (*tsf_writer_perpav)(scalars_avg);
    progress++;
  }
}