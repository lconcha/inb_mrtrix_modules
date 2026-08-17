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

#include <stdio.h>
#include <vector> 


using namespace MR;
using namespace App;

using Fixel::index_type;


#define DEFAULT_ANGULAR_THRESHOLD 45.0
#define DEFAULT_NOFIXEL_VALUE -1.0


void usage ()
{

  AUTHOR = "Original implementation by David Raffelt (david.raffelt@florey.edu.au), "
           "Hacked by Luis Concha (lconcha@unam.mx) in January-August 2024";

  SYNOPSIS = "Map corresponding fixel indices to a track scalar file based on an input tractogram";

  DESCRIPTION
  + "This command identifies, for each streamline point, the index of the underlying fixel that is most parallel with the streamline segment."
    "The outputs are .tsf files indicating the indices of the most parallel fixel, the fixel values of the most parallel and most perpendicular fixels,"
    "as well as the average value of all fixels except the parallel one (i.e., the average perpendicular value).\n\n"
    "NOTE that a value of -1 (minus one) will be written to the tsf files in any point where there is no fixel value "
    "(this can happen if there is no fixels in that position, or for perpendicular values if nfixels=1 in that position).";

  ARGUMENTS
  + Argument ("fixel_in", "the input fixel data file (within the fixel directory)").type_image_in ()
  + Argument ("tracks",   "the input track file ").type_tracks_in ()
  + Argument ("tsf_fixel_indices",      "the output track scalar file indicating the index of the fixel that is most parallel to the streamline segment").type_file_out ()
  + Argument ("tsf_values_par", "the output file containing the metric corresponding to the fixel most parallel to the streamline segment").type_file_out ()
  + Argument ("tsf_values_perp", "the output file containing the metric corresponding to the fixel most perpendicular to the streamline segment").type_file_out ()
  + Argument ("tsf_values_perpav", "the output file containing the average metric from all fixels except the most parallel one").type_file_out ();

  OPTIONS
  + Option ("angle", "the max anglular threshold for computing correspondence "
                     "between a fixel direction and track tangent "
                     "(default = " + str(DEFAULT_ANGULAR_THRESHOLD, 2) + " degrees)")
  + Argument ("value").type_float (0.001, 90.0)
  + Option ("nofixel_value", "the value to write in the track scalar file when no fixel is found at a streamline point (default = " + str(DEFAULT_NOFIXEL_VALUE, 2) + ")")
  + Argument ("value").type_float ();
}

using SetVoxelDir = DWI::Tractography::Mapping::SetVoxelDir;




void run ()
{
  auto in_data_image = Fixel::open_fixel_data_file<float> (argument[0]);
  // auto in_data_image = Fixel::open_fixel_data_file<float> (f_directions);
  if (in_data_image.size(2) != 1)
    throw Exception ("Only a single scalar value for each fixel can be output as a track scalar file, "
                     "therefore the input fixel data file must have dimension Nx1x1");

  Header in_index_header = Fixel::find_index_header (Fixel::get_fixel_directory (argument[0]));
  auto in_index_image = in_index_header.get_image<index_type>();
  auto in_directions_image = Fixel::find_directions_header (Fixel::get_fixel_directory (argument[0])).get_image<float>().with_direct_io();

  DWI::Tractography::Properties properties;
  DWI::Tractography::Reader<float> reader (argument[1], properties);
  properties.comments.push_back ("Created using tcksamplefixels");
  properties.comments.push_back ("Source fixel image: " + Path::basename (argument[0]));
  properties.comments.push_back ("Source track file: " + Path::basename (argument[1]));


  DWI::Tractography::ScalarWriter<float> tsf_writer_fixelids     (argument[2], properties);
  DWI::Tractography::ScalarWriter<float> tsf_writer_value_par    (argument[3], properties);
  DWI::Tractography::ScalarWriter<float> tsf_writer_value_perp   (argument[4], properties);
  DWI::Tractography::ScalarWriter<float> tsf_writer_value_perpav (argument[5], properties);


  float angular_threshold = get_option_value ("angle", DEFAULT_ANGULAR_THRESHOLD);
  const float angular_threshold_dp = cos (angular_threshold * (Math::pi / 180.0));

  float nofixel_value = get_option_value ("nofixel_value", DEFAULT_NOFIXEL_VALUE);

  const size_t num_tracks = properties["count"].empty() ? 0 : to<int> (properties["count"]);

  DWI::Tractography::Mapping::TrackMapperBase mapper (in_index_image);
  mapper.set_use_precise_mapping (true);

  ProgressBar progress ("mapping dot products to streamline points", int(num_tracks));
  DWI::Tractography::Streamline<float> tck;
  DWI::Tractography::TrackScalar<float> fixelids;
  DWI::Tractography::TrackScalar<float> values_par;
  DWI::Tractography::TrackScalar<float> values_perp;
  DWI::Tractography::TrackScalar<float> values_perpav;
  
  const Transform transform (in_index_image);
  Eigen::Vector3d voxel_pos;

  int streamline_index = 0;
  while (reader (tck)) {
    SetVoxelDir dixels;
    mapper (tck, dixels);
    fixelids.clear();
    fixelids.set_index (tck.get_index());
    fixelids.resize (tck.size(), 0.0f);
    values_par.clear();
    values_par.set_index (tck.get_index());
    values_par.resize (tck.size(), 0.0f);
    values_perp.clear();
    values_perp.set_index (tck.get_index());
    values_perp.resize (tck.size(), 0.0f);
    values_perpav.clear();
    values_perpav.set_index (tck.get_index());
    values_perpav.resize (tck.size(), 0.0f);
    for (size_t p = 0; p < tck.size(); ++p) {
      voxel_pos = transform.scanner2voxel * tck[p].cast<default_type> ();
      for (SetVoxelDir::const_iterator d = dixels.begin(); d != dixels.end(); ++d) {
        if ((int)round(voxel_pos[0]) == (*d)[0] && (int)round(voxel_pos[1]) == (*d)[1] && (int)round(voxel_pos[2]) == (*d)[2]) {
          assign_pos_of (*d).to (in_index_image);
          Eigen::Vector3f dir = d->get_dir().cast<float>();
          dir.normalize();
          float largest_dp = 0.0f;
          float lowest_dp  = 1.0f;
          float value_par  = nofixel_value;
          float value_perp = nofixel_value;
          float value_perpav = nofixel_value;
          int32_t closest_fixel_index = nofixel_value;
          int32_t farthest_fixel_index = nofixel_value;

          in_index_image.index(3) = 0;
          index_type num_fixels_in_voxel = in_index_image.value();
          in_index_image.index(3) = 1;
          index_type offset = in_index_image.value();

          DEBUG("Streamline " + str(streamline_index) + " (of " + str(num_tracks) + "), Point " + str(p) + " (of " + str(tck.size()) + "), nfixels " + str(num_fixels_in_voxel) + ", segment vector is [" + str(dir[0],2) + " " + str(dir[1],2) + " " + str(dir[2],2) + "]");
          //std::printf("  Position (x,y,z): %1.2f, %1.2f, %1.2f\n", voxel_pos[0],voxel_pos[1],voxel_pos[2]);

          //std::fprintf(stdout,"value_par is  %1.2f, value_perp is %1.2f, value_perpav is %1.2f\n",value_par,value_perp,value_perpav);

          if ( num_fixels_in_voxel < 1 ){
            WARN("No fixels exist in streamline " + str(streamline_index) + " point " + str(p) + " Position (x,y,z): " + str(voxel_pos[0],1) + ", " + str(voxel_pos[1],1) + ", " + str(voxel_pos[2],1));
            DEBUG("    Most parallel fixel index :     " + str(int(nofixel_value)));
            DEBUG("    Most perpendicular fixel index: " + str(int(nofixel_value)));
            DEBUG("    Parallel fixel value :          " + str(nofixel_value));
            DEBUG("    Perpendicular fixel value :     " + str(nofixel_value));
            DEBUG("    Perpendicular_av fixel value :  " + str(nofixel_value));
            fixelids[p]      = nofixel_value;
            values_par[p]    = nofixel_value;
            values_perp[p]   = nofixel_value;
            values_perpav[p] = nofixel_value;
            continue;
          }
          
          std::vector<float> fixel_values;
          
          for (size_t fixel = 0; fixel < num_fixels_in_voxel; ++fixel) {
            in_directions_image.index(0) = offset + fixel;
            const float dp = abs (dir.dot (Eigen::Vector3f (in_directions_image.row(1))));
            in_data_image.index(0) = offset + fixel;
            const float value = in_data_image.value();
            DEBUG("  fixel " + str(int(fixel)) + ", vector is [" + str(in_directions_image.row(1)[0],2) + "\t" + str(in_directions_image.row(1)[1],2) + "\t" + str(in_directions_image.row(1)[2],2) + "]\tdp is " + str(dp,2) + ", value is " + str(float(value),2) + ")");
            fixel_values.push_back(value);
            if (dp > largest_dp) {
              largest_dp = dp;
              closest_fixel_index = fixel;
            }
            if (dp < lowest_dp) {
              lowest_dp = dp;
              farthest_fixel_index = fixel;
            }
          }
          const bool is_sufficiently_parallel = (largest_dp >= angular_threshold_dp);
          if (!is_sufficiently_parallel) {
              DEBUG("largest_dp " + str(largest_dp,2) + " is lower than angular_threshold_dp " + str(angular_threshold_dp,2) + " (in streamline " + str(streamline_index) + " point " + str(p) + ") ");
              closest_fixel_index = -1;
          }

          if (closest_fixel_index < 0) {
            DEBUG("No parallel fixel assigned here: Position (x,y,z): " + str(voxel_pos[0],1) + ", " + str(voxel_pos[1],1) + ", " + str(voxel_pos[2],1));
            fixelids[p]       = nofixel_value;
            value_par         = nofixel_value;
          } else {
            value_par  = fixel_values[closest_fixel_index];
          }

            if (fixel_values.size() == 1) {
              // only one fixel in the voxel: it can only play one role at a time
              if (is_sufficiently_parallel) {
                value_perp   = nofixel_value;
                value_perpav = nofixel_value;
              } else {
                value_perp   = fixel_values[0];
                value_perpav = fixel_values[0];
              }
            } else {
              value_perp = fixel_values[farthest_fixel_index];
              const float sum_all = accumulate(fixel_values.begin(), fixel_values.end(), 0.0f);
              if (closest_fixel_index >= 0) {
                // exclude the most-parallel fixel, since it was validly assigned to value_par
                value_perpav = (sum_all - fixel_values[closest_fixel_index]) / (fixel_values.size() - 1);
              } else {
                // no fixel qualified as "most parallel", so nothing to exclude
                value_perpav = sum_all / fixel_values.size();
              }
            }
            DEBUG("    Most parallel fixel index :     " + str(int(closest_fixel_index)));
            DEBUG("    Most perpendicular fixel index: " + str(int(farthest_fixel_index)));
            DEBUG("    Parallel fixel value :          " + str(value_par,2));
            DEBUG("    Perpendicular fixel value :     " + str(value_perp,2));
            DEBUG("    Perpendicular_av fixel value :  " + str(value_perpav,2));
            
            fixelids[p]      = float(closest_fixel_index);
            values_par[p]    = value_par;
            values_perp[p]   = value_perp;
            values_perpav[p] = value_perpav;
        }
      }
    }
    tsf_writer_fixelids     (fixelids);
    tsf_writer_value_par    (values_par);
    tsf_writer_value_perp   (values_perp);
    tsf_writer_value_perpav (values_perpav);
    progress++;
    streamline_index = streamline_index +1;
    
  }
}