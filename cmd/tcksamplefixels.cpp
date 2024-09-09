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



void usage ()
{

  AUTHOR = "Original implementation by David Raffelt (david.raffelt@florey.edu.au), "
           "Hacked by Luis Concha (lconcha@unam.mx) in January-August 2024";

  SYNOPSIS = "Map corresponding fixel indices to a track scalar file based on an input tractogram";

  DESCRIPTION
  + "This command identifies, for each streamline point, the index of the underlying fixel that is most parallel with the streamline segment."
    "The output is a .tsf file indicating fixel indices at each point.";

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
  + Argument ("value").type_float (0.001, 90.0);

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

  const size_t num_tracks = properties["count"].empty() ? 0 : to<int> (properties["count"]);

  DWI::Tractography::Mapping::TrackMapperBase mapper (in_index_image);
  mapper.set_use_precise_mapping (true);

  ProgressBar progress ("mapping dot products to streamline points", num_tracks);
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
          float value_par  = -1;
          float value_perp = -1;
          float value_perpav = -1;
          int32_t closest_fixel_index = -1;
          int32_t farthest_fixel_index = -1;

          in_index_image.index(3) = 0;
          index_type num_fixels_in_voxel = in_index_image.value();
          in_index_image.index(3) = 1;
          index_type offset = in_index_image.value();

          //std::printf("Streamline %d, Point %d, nfixels %d, vector is [%1.2f %1.2f %1.2f]",streamline_index,int(p), int(num_fixels_in_voxel), dir[0], dir[1], dir[2] );
          //std::printf("  Position (x,y,z): %1.2f, %1.2f, %1.2f\n", voxel_pos[0],voxel_pos[1],voxel_pos[2]);

          if ( num_fixels_in_voxel < 1 ){
            std::fprintf(stderr," No fixels exist in streamline %d point %d Position (x,y,z): %1.2f, %1.2f, %1.2f\n", streamline_index, int(p), voxel_pos[0],voxel_pos[1],voxel_pos[2]);
          }
          
          std::vector<float> fixel_values;
          
          for (size_t fixel = 0; fixel < num_fixels_in_voxel; ++fixel) {
            in_directions_image.index(0) = offset + fixel;
            const float dp = abs (dir.dot (Eigen::Vector3f (in_directions_image.row(1))));
            in_data_image.index(0) = offset + fixel;
            const float value = in_data_image.value();
            //std::fprintf(stdout,"  fixel %d, vector is [%1.2f\t%1.2f\t%1.2f]\tdp is %1.2f, value is %1.4f\n", int(fixel), in_directions_image.row(1)[0],in_directions_image.row(1)[1],in_directions_image.row(1)[2], dp, float(value));
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
          if (largest_dp < angular_threshold_dp) {
              std::fprintf(stderr,"  largest_dp %g is lower than angular_threshold_dp %g\n",largest_dp,angular_threshold_dp);
              closest_fixel_index = -1;
          }

          if (closest_fixel_index < 0) {
            std::fprintf(stderr,"    ---- No fixel assigned here:");
            std::fprintf(stderr,"  Position (x,y,z): %1.2f, %1.2f, %1.2f\n", voxel_pos[0],voxel_pos[1],voxel_pos[2]);
          } else {
            value_par  = fixel_values[closest_fixel_index];
            value_perp = fixel_values[farthest_fixel_index];
            //std::printf("    Fixel %d assigned as parallel, with a dot product of %1.2f and value of %1.4f\n", int(closest_fixel_index), largest_dp, value_par );
            
            if (num_fixels_in_voxel < 2){
              value_perp = -1;
            } else {
              //std::printf("    Fixel %d is the most perpendicular, with a dot product of %1.2f and value of %1.4f\n", int(farthest_fixel_index), lowest_dp, value_perp );
              fixel_values.erase(fixel_values.begin()+int(closest_fixel_index));//remove par value
              value_perpav = accumulate(fixel_values.begin(), fixel_values.end(),0.0) / fixel_values.size();
            }
            //std::printf("    Parallel fixel value :          %1.3f\n",value_par);
            //std::printf("    Perpendicular fixel value :     %1.3f\n",value_perp);
            //std::printf("    Perpendicular_av fixel value :  %1.3f\n",value_perpav);
            
            fixelids[p] = float(closest_fixel_index);
            values_par[p] = value_par;
            values_perp[p] = value_perp;
            values_perpav[p] = value_perpav;            
          }
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

