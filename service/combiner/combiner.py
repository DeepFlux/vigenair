# Copyright 2024 Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Vigenair Combiner service.

This module provides functionality to combine individual cuts of the input video
based on user-specific rendering settings.
"""

import dataclasses
import gc
import json
import logging
import os
import pathlib
import re
import sys
import subprocess
import tempfile
from typing import Any, Dict, Optional, Sequence, Tuple, Union
from urllib import parse

import config as ConfigService
import pandas as pd
import storage as StorageService
import utils as Utils
import vertexai
from vertexai.generative_models import GenerativeModel, Part


@dataclasses.dataclass(init=False)
class VideoVariantRenderSettings:
  """Represents the settings for a video variant.

  Attributes:
    generate_image_assets: Whether to generate image assets.
    generate_text_assets: Whether to generate text assets.
    formats: Which formats to render (horizontal, square, vertical).
    use_music_overlay: Whether to use the music overlay feature, where a
      contiguous section of the input's background music will be used for the
      video variant instead of the individual segments' background music.
    use_continuous_audio: Whether to use a contiguous section of the input's
      audio track for the video variant instead of the individual segments'
      audio track portions.
    fade_out: Whether to fade out the end of the video variant.
    overlay_type: How to overlay music / audio for the variant.
  """

  generate_image_assets: bool = False
  generate_text_assets: bool = False
  formats: Sequence[Utils.RenderFormatType] = None
  use_music_overlay: bool = False
  use_continuous_audio: bool = False
  fade_out: bool = False
  overlay_type: Utils.RenderOverlayType = None

  def __init__(self, **kwargs):
    field_names = set([f.name for f in dataclasses.fields(self)])
    for k, v in kwargs.items():
      if k in field_names:
        setattr(self, k, v)

  def __str__(self):
    return (
        'VideoVariantRenderSettings('
        f'generate_image_assets={self.generate_image_assets}, '
        f'generate_text_assets={self.generate_text_assets}, '
        f'formats={self.formats}, '
        f'use_music_overlay={self.use_music_overlay}, '
        f'use_continuous_audio={self.use_continuous_audio}, '
        f'fade_out={self.fade_out}, '
        f'overlay_type={self.overlay_type})'
    )


@dataclasses.dataclass(init=False)
class VideoVariantSegment:
  """Represents a segment of a video variant.

  Attributes:
    av_segment_id: The id of the A/V segment.
    start_s: The start time of the A/V segment in seconds.
    end_s: The end time of the A/V segment in seconds.
  """

  av_segment_id: int
  start_s: float
  end_s: float

  def __init__(self, **kwargs):
    field_names = set([f.name for f in dataclasses.fields(self)])
    for k, v in kwargs.items():
      if k in field_names:
        setattr(self, k, v)

  def __str__(self):
    return (
        f'VideoVariantSegment(av_segment_id={self.av_segment_id}, '
        f'start_s={self.start_s}, '
        f'end_s={self.end_s})'
    )


@dataclasses.dataclass(init=False)
class VideoVariant:
  """Represents a video variant.

  Attributes:
    variant_id: The id of the variant.
    av_segments: The A/V segments of the variant, mapped by segment id.
    title: The title of the variant.
    description: The description of the variant.
    score: The score of the variant.
    score_reasoning: The reasoning behind the score of the variant.
    render_settings: The render settings of the variant.
  """

  variant_id: int
  av_segments: Dict[str, VideoVariantSegment]
  title: str
  description: str
  score: float
  score_reasoning: str
  render_settings: VideoVariantRenderSettings

  def __init__(self, **kwargs):
    field_names = set([f.name for f in dataclasses.fields(self)])
    for k, v in kwargs.items():
      if k in field_names:
        setattr(self, k, v)

  def __str__(self):
    return (
        f'VideoVariant(variant_id={self.variant_id}, '
        f'av_segments={self.av_segments}, '
        f'title={self.title}, '
        f'description={self.description}, '
        f'score={self.score}, '
        f'score_reasoning={self.score_reasoning}, '
        f'render_settings={self.render_settings})'
    )


class Combiner:
  """Encapsulates all the combination logic."""

  def __init__(self, gcs_bucket_name: str, render_file: Utils.TriggerFile):
    """Initialiser.

    Args:
      gcs_bucket_name: The GCS bucket to read from and store files in.
      render_file: Path to the input rendering file, which is in a
        `<timestamp>-combos` subdirectory of the root video folder (see
        `extractor.Extractor` for more information on root folder naming).
    """
    self.gcs_bucket_name = gcs_bucket_name
    self.render_file = render_file
    vertexai.init(
        project=ConfigService.GCP_PROJECT_ID,
        location=ConfigService.GCP_LOCATION,
    )
    self.text_model = GenerativeModel(ConfigService.CONFIG_TEXT_MODEL)
    self.vision_model = GenerativeModel(ConfigService.CONFIG_VISION_MODEL)

  def check_finalise_render(self, variants_count: int):
    """Checks whether all variants have been rendered to trigger `finalise`."""
    rendered_count = len(
        StorageService.filter_files(
            bucket_name=self.gcs_bucket_name,
            prefix=f'{self.render_file.gcs_folder}/',
            suffix=ConfigService.OUTPUT_COMBINATIONS_FILE,
        )
    )
    if rendered_count == variants_count:
      finalise_file_path = (
          f'{variants_count}-{variants_count}_'
          f'{ConfigService.INPUT_RENDERING_FINALISE_FILE}'
      )
      with open(finalise_file_path, 'w', encoding='utf8'):
        pass

      StorageService.upload_gcs_file(
          file_path=finalise_file_path,
          bucket_name=self.gcs_bucket_name,
          destination_file_name=str(
              pathlib.Path(self.render_file.gcs_folder, finalise_file_path)
          ),
      )

  def finalise_render(self):
    """Combines all generated <id>_combos.json into a single one."""
    logging.info('COMBINER - Finalising rendering...')
    tmp_dir = tempfile.mkdtemp()
    render_output_dicts = [
        json.loads(json_file_contents.decode('utf-8'))
        for json_file_contents in StorageService.filter_files(
            bucket_name=self.gcs_bucket_name,
            prefix=f'{self.render_file.gcs_folder}/',
            suffix=ConfigService.OUTPUT_COMBINATIONS_FILE,
            fetch_content=True,
        )
    ]
    output = {}
    for render_output_dict in render_output_dicts:
      for k, v in render_output_dict.items():
        output[k] = v

    combos_json_path = os.path.join(
        tmp_dir,
        ConfigService.OUTPUT_COMBINATIONS_FILE,
    )
    with open(combos_json_path, 'w', encoding='utf8') as f:
      json.dump(output, f, indent=2)

    StorageService.upload_gcs_file(
        file_path=combos_json_path,
        bucket_name=self.gcs_bucket_name,
        destination_file_name=str(
            pathlib.Path(
                self.render_file.gcs_folder,
                ConfigService.OUTPUT_COMBINATIONS_FILE,
            )
        ),
    )
    gc.collect()
    logging.info('COMBINER - Rendering completed successfully!')

  def render(self):
    """Renders a single video based on the input rendering settings."""
    variant_id = self.render_file.file_name.split('_')[0]
    logging.info('COMBINER - Starting rendering variant %s...', variant_id)
    tmp_dir = tempfile.mkdtemp()
    root_video_folder = self.render_file.gcs_root_folder
    video_file_name = next(
        iter(
            StorageService.filter_video_files(
                prefix=f'{root_video_folder}/{ConfigService.INPUT_FILENAME}',
                bucket_name=self.gcs_bucket_name,
                first_only=True,
            )
        ), None
    )
    logging.info('RENDERING - Video file name: %s', video_file_name)
    video_file_path = StorageService.download_gcs_file(
        file_path=Utils.TriggerFile(video_file_name),
        output_dir=tmp_dir,
        bucket_name=self.gcs_bucket_name,
    )
    logging.info('RENDERING - Video file path: %s', video_file_path)
    _, video_ext = os.path.splitext(video_file_path)
    has_audio = StorageService.download_gcs_file(
        file_path=Utils.TriggerFile(
            str(
                pathlib.Path(
                    root_video_folder, f'{ConfigService.INPUT_FILENAME}.wav'
                )
            )
        ),
        output_dir=tmp_dir,
        bucket_name=self.gcs_bucket_name,
    ) is not None
    logging.info('RENDERING - Video has audio track? %s', has_audio)
    speech_track_path = StorageService.download_gcs_file(
        file_path=Utils.TriggerFile(
            str(
                pathlib.Path(
                    root_video_folder, ConfigService.OUTPUT_SPEECH_FILE
                )
            )
        ),
        output_dir=tmp_dir,
        bucket_name=self.gcs_bucket_name,
    )
    logging.info('RENDERING - Speech track path: %s', speech_track_path)
    music_track_path = StorageService.download_gcs_file(
        file_path=Utils.TriggerFile(
            str(
                pathlib.Path(
                    root_video_folder, ConfigService.OUTPUT_MUSIC_FILE
                )
            )
        ),
        output_dir=tmp_dir,
        bucket_name=self.gcs_bucket_name,
    )
    logging.info('RENDERING - Music track path: %s', music_track_path)
    video_language = StorageService.download_gcs_file(
        file_path=Utils.TriggerFile(
            str(
                pathlib.
                Path(root_video_folder, ConfigService.OUTPUT_LANGUAGE_FILE)
            )
        ),
        output_dir=tmp_dir,
        bucket_name=self.gcs_bucket_name,
        fetch_contents=True,
    ) or ConfigService.DEFAULT_VIDEO_LANGUAGE
    logging.info('RENDERING - Video language: %s', video_language)
    square_video_file_path = StorageService.download_gcs_file(
        file_path=Utils.TriggerFile(
            str(
                pathlib.Path(
                    self.render_file.gcs_folder,
                    ConfigService.INPUT_SQUARE_CROP_FILE.replace(
                        '.txt', video_ext
                    )
                )
            )
        ),
        output_dir=tmp_dir,
        bucket_name=self.gcs_bucket_name,
    )
    logging.info(
        'RENDERING - Square video file path: %s', square_video_file_path
    )
    vertical_video_file_path = StorageService.download_gcs_file(
        file_path=Utils.TriggerFile(
            str(
                pathlib.Path(
                    self.render_file.gcs_folder,
                    ConfigService.INPUT_VERTICAL_CROP_FILE.replace(
                        '.txt', video_ext
                    )
                )
            )
        ),
        output_dir=tmp_dir,
        bucket_name=self.gcs_bucket_name,
    )
    logging.info(
        'RENDERING - Vertical video file path: %s', vertical_video_file_path
    )
    render_file_contents = StorageService.download_gcs_file(
        file_path=self.render_file,
        bucket_name=self.gcs_bucket_name,
        fetch_contents=True,
    )
    video_variant = list(
        map(
            _video_variant_mapper,
            enumerate(json.loads(render_file_contents.decode('utf-8'))),
        )
    )[0]
    combos_dir = tempfile.mkdtemp()
    rendered_combos = {}
    rendered_variant_paths = _render_video_variant(
        output_dir=combos_dir,
        gcs_folder_path=self.render_file.gcs_folder,
        gcs_bucket_name=self.gcs_bucket_name,
        video_file_path=video_file_path,
        square_video_file_path=square_video_file_path,
        vertical_video_file_path=vertical_video_file_path,
        has_audio=has_audio,
        speech_track_path=speech_track_path,
        music_track_path=music_track_path,
        video_variant=video_variant,
        vision_model=self.vision_model,
        video_language=video_language,
    )
    combo = dataclasses.asdict(video_variant)
    combo.update(rendered_variant_paths)
    combo['av_segments'] = {
        f'_{segment_id}': segment
        for segment_id, segment in combo['av_segments'].items()
    }
    rendered_combos[f'_{video_variant.variant_id}'] = combo
    logging.info(
        'RENDERING - Rendered variant as: %r',
        rendered_combos,
    )
    combos_json_path = os.path.join(
        combos_dir,
        f'{variant_id}_{ConfigService.OUTPUT_COMBINATIONS_FILE}',
    )
    with open(combos_json_path, 'w', encoding='utf8') as f:
      json.dump(rendered_combos, f, indent=2)

    StorageService.upload_gcs_dir(
        source_directory=combos_dir,
        bucket_name=self.gcs_bucket_name,
        target_dir=self.render_file.gcs_folder,
    )

    self.check_finalise_render(variants_count=int(variant_id.split('-')[1]))
    gc.collect()
    logging.info(
        'COMBINER - Rendering variant %s completed successfully!',
        variant_id,
    )

  def initial_render(self):
    """Renders videos based on the input rendering settings."""
    logging.info('COMBINER - Starting rendering...')
    tmp_dir = tempfile.mkdtemp()
    root_video_folder = self.render_file.gcs_root_folder
    video_file_name = next(
        iter(
            StorageService.filter_video_files(
                prefix=f'{root_video_folder}/{ConfigService.INPUT_FILENAME}',
                bucket_name=self.gcs_bucket_name,
                first_only=True,
            )
        ), None
    )
    logging.info('RENDERING - Video file name: %s', video_file_name)
    video_file_path = StorageService.download_gcs_file(
        file_path=Utils.TriggerFile(video_file_name),
        output_dir=tmp_dir,
        bucket_name=self.gcs_bucket_name,
    )
    logging.info('RENDERING - Video file path: %s', video_file_path)
    square_crop_file_path = StorageService.download_gcs_file(
        file_path=Utils.TriggerFile(
            str(
                pathlib.Path(
                    self.render_file.gcs_folder,
                    ConfigService.INPUT_SQUARE_CROP_FILE
                )
            )
        ),
        output_dir=tmp_dir,
        bucket_name=self.gcs_bucket_name,
    )
    logging.info('RENDERING - Square crop commands: %s', square_crop_file_path)
    vertical_crop_file_path = StorageService.download_gcs_file(
        file_path=Utils.TriggerFile(
            str(
                pathlib.Path(
                    self.render_file.gcs_folder,
                    ConfigService.INPUT_VERTICAL_CROP_FILE
                )
            )
        ),
        output_dir=tmp_dir,
        bucket_name=self.gcs_bucket_name,
    )
    logging.info(
        'RENDERING - Vertical crop commands: %s', vertical_crop_file_path
    )
    render_file_contents = StorageService.download_gcs_file(
        file_path=self.render_file,
        bucket_name=self.gcs_bucket_name,
        fetch_contents=True,
    )
    video_variants = list(
        map(
            _video_variant_mapper,
            enumerate(json.loads(render_file_contents.decode('utf-8'))),
        )
    )
    logging.info(
        'RENDERING - Rendering %d video variants: %r',
        len(video_variants),
        video_variants,
    )
    combos_dir = tempfile.mkdtemp()
    _create_cropped_videos(
        video_variants=video_variants,
        video_file_path=video_file_path,
        square_crop_file_path=square_crop_file_path,
        vertical_crop_file_path=vertical_crop_file_path,
        output_dir=combos_dir,
    )
    StorageService.upload_gcs_dir(
        source_directory=combos_dir,
        bucket_name=self.gcs_bucket_name,
        target_dir=self.render_file.gcs_folder,
    )
    for video_variant in video_variants:
      variant_destination_file_path = (
          f'{video_variant.variant_id}-{len(video_variants)}'
          f'_{ConfigService.INPUT_RENDERING_FILE}'
      )
      variant_dict = dataclasses.asdict(video_variant)
      variant_dict['av_segments'] = [
          s for s in variant_dict['av_segments'].values()
      ]
      variant_json_path = os.path.join(
          combos_dir,
          variant_destination_file_path,
      )
      with open(variant_json_path, 'w', encoding='utf8') as f:
        json.dump([variant_dict], f, indent=2)

      StorageService.upload_gcs_file(
          file_path=variant_json_path,
          bucket_name=self.gcs_bucket_name,
          destination_file_name=str(
              pathlib.Path(
                  self.render_file.gcs_folder,
                  variant_destination_file_path,
              )
          ),
      )
    gc.collect()
    logging.info('COMBINER - Initial render completed successfully!')


def _video_variant_mapper(index_variant_dict_tuple: Tuple[int, Dict[str, Any]]):
  index, variant_dict = index_variant_dict_tuple
  segment_dicts = variant_dict.pop('av_segments', None)
  render_settings_dict = variant_dict.pop('render_settings', None)
  segments = {
      str(segment_dict['av_segment_id']): VideoVariantSegment(**segment_dict)
      for segment_dict in segment_dicts
  }
  video_variant_settings = VideoVariantRenderSettings(**render_settings_dict)
  variant_id = variant_dict.pop('variant_id', index)

  return VideoVariant(
      variant_id=variant_id,
      av_segments=segments,
      render_settings=video_variant_settings,
      **variant_dict,
  )


def _create_cropped_videos(
    video_variants: Sequence[VideoVariant],
    video_file_path: str,
    square_crop_file_path: Optional[str],
    vertical_crop_file_path: Optional[str],
    output_dir: str,
) -> Tuple[Optional[str], Optional[str]]:
  """Creates cropped videos for the given video variants.

  Args:
    video_variants: The video variants to create cropped videos for.
    video_file_path: The path to the input video file.
    square_crop_file_path: The square crop commands file, or None.
    vertical_crop_file_path: The vertical crop commands file, or None.
    output_dir: The output directory to use.

  Returns:
    The paths to the square and vertical cropped videos.
  """
  square_video_file_path = None
  vertical_video_file_path = None
  _, video_ext = os.path.splitext(video_file_path)

  if bool(
      list(
          filter(
              lambda variant: Utils.RenderFormatType.SQUARE.value in variant.
              render_settings.formats, video_variants
          )
      )
  ):
    square_video_file_path = _create_cropped_video(
        video_file_path=video_file_path,
        crop_file_path=square_crop_file_path,
        output_dir=output_dir,
        format_type=Utils.RenderFormatType.SQUARE.value,
        video_ext=video_ext,
    )
  if bool(
      list(
          filter(
              lambda variant: Utils.RenderFormatType.VERTICAL.value in variant.
              render_settings.formats, video_variants
          )
      )
  ):
    vertical_video_file_path = _create_cropped_video(
        video_file_path=video_file_path,
        crop_file_path=vertical_crop_file_path,
        output_dir=output_dir,
        format_type=Utils.RenderFormatType.VERTICAL.value,
        video_ext=video_ext,
    )

  return square_video_file_path, vertical_video_file_path


def _create_cropped_video(
    video_file_path: str,
    crop_file_path: Optional[str],
    output_dir: str,
    format_type: str,
    video_ext: str,
) -> Optional[str]:
  """Creates a cropped video for the given format.

  Args:
    video_file_path: The path to the input video file.
    crop_file_path: The crop commands file, or None.
    output_dir: The output directory to use.
    format_type: The format to create a cropped video for.
    video_ext: The extension of the video file.

  Returns:
    The paths to the cropped video, or None.
  """
  cropped_video_path = None
  if crop_file_path:
    with open(crop_file_path, mode='r', encoding='utf8') as f:
      first_line = f.readline().strip()
    matches = re.search(r'crop w (.*), crop h (.*);', first_line)

    w = matches.group(1) if matches else None
    h = matches.group(2) if matches else None

    cropped_video_path = str(
        pathlib.Path(output_dir, f'{format_type}{video_ext}')
    )
    Utils.execute_subprocess_commands(
        cmds=[
            'ffmpeg',
            '-i',
            video_file_path,
            '-filter_complex',
            f'[0:v]sendcmd=f={crop_file_path},crop[cropped];'
            f'[cropped]crop={w}:{h}',
            cropped_video_path,
        ],
        description=(f'render full {format_type} format using ffmpeg'),
    )
  return cropped_video_path


def _render_video_variant(
    output_dir: str,
    gcs_folder_path: str,
    gcs_bucket_name: str,
    video_file_path: str,
    square_video_file_path: Optional[str],
    vertical_video_file_path: Optional[str],
    has_audio: bool,
    speech_track_path: Optional[str],
    music_track_path: Optional[str],
    video_variant: VideoVariant,
    vision_model: GenerativeModel,
    video_language: str,
) -> Dict[str, str]:
  """Renders a video variant in all formats.

  Args:
    output_dir: The output directory to use.
    gcs_folder_path: The GCS folder path to use.
    gcs_bucket_name: The GCS bucket name to upload to.
    video_file_path: The path to the input video file.
    square_video_file_path: The path to the square crop of the input video file.
    vertical_video_file_path: The path to the vertical crop of the input video
      file.
    has_audio: Whether the video has an audio track.
    speech_track_path: The path to the video's speech track, or None.
    music_track_path: The path to the video's music track, or None.
    video_variant: The video variant to be rendered.
    vision_model: The vision model to use.
    video_language: The video language.

  Returns:
    The rendered paths keyed by the format type.
  """
  logging.info('RENDERING - Rendering video variant: %s', video_variant)
  _, video_ext = os.path.splitext(video_file_path)
  shot_groups = _group_consecutive_segments(
      list(video_variant.av_segments.keys())
  )
  shot_timestamps = list(
      map(
          lambda group: (
              video_variant.av_segments[group[0]].start_s,
              video_variant.av_segments[group[1]].end_s,
          ),
          shot_groups,
      )
  )
  video_duration = Utils.get_media_duration(video_file_path)
  (
      full_av_select_filter,
      music_overlay_select_filter,
      continuous_audio_select_filter,
  ) = _build_ffmpeg_filters(
      shot_timestamps,
      has_audio,
      video_variant.render_settings,
      video_duration,
  )

  ffmpeg_cmds = _get_variant_ffmpeg_commands(
      video_file_path=video_file_path,
      speech_track_path=speech_track_path,
      music_track_path=music_track_path,
      has_audio=has_audio,
      music_overlay=video_variant.render_settings.use_music_overlay,
      continuous_audio=video_variant.render_settings.use_continuous_audio,
      full_av_select_filter=full_av_select_filter,
      music_overlay_select_filter=music_overlay_select_filter,
      continuous_audio_select_filter=continuous_audio_select_filter,
  )

  horizontal_combo_name = f'combo_{video_variant.variant_id}_h{video_ext}'
  horizontal_combo_path = str(pathlib.Path(output_dir, horizontal_combo_name))
  ffmpeg_cmds.append(horizontal_combo_path)

  Utils.execute_subprocess_commands(
      cmds=ffmpeg_cmds,
      description=(
          'render horizontal variant with id '
          f'{video_variant.variant_id} using ffmpeg'
      ),
  )
  rendered_paths = {
      Utils.RenderFormatType.HORIZONTAL.value: {
          'path': horizontal_combo_name
      }
  }
  if video_variant.render_settings.generate_image_assets:
    StorageService.upload_gcs_dir(
        source_directory=output_dir,
        bucket_name=gcs_bucket_name,
        target_dir=gcs_folder_path,
    )
    assets = _generate_image_assets(
        vision_model=vision_model,
        video_file_path=horizontal_combo_path,
        gcs_bucket_name=gcs_bucket_name,
        gcs_folder_path=gcs_folder_path,
        output_path=output_dir,
        variant_id=video_variant.variant_id,
        format_type=Utils.RenderFormatType.HORIZONTAL.value,
    )
    if assets:
      rendered_paths[Utils.RenderFormatType.HORIZONTAL.value]['images'] = assets

  formats_to_render = {}
  if (
      Utils.RenderFormatType.SQUARE.value
      in video_variant.render_settings.formats
  ):
    formats_to_render[Utils.RenderFormatType.SQUARE.value] = {
        'blur_filter': ConfigService.FFMPEG_SQUARE_BLUR_FILTER,
        'crop_file_path': square_video_file_path
    }
  if (
      Utils.RenderFormatType.VERTICAL.value
      in video_variant.render_settings.formats
  ):
    formats_to_render[Utils.RenderFormatType.VERTICAL.value] = {
        'blur_filter': ConfigService.FFMPEG_VERTICAL_BLUR_FILTER,
        'crop_file_path': vertical_video_file_path
    }
  for format_type, format_instructions in formats_to_render.items():
    format_ffmpeg_cmds = None
    if format_instructions['crop_file_path']:
      format_ffmpeg_cmds = _get_variant_ffmpeg_commands(
          video_file_path=format_instructions['crop_file_path'],
          speech_track_path=speech_track_path,
          music_track_path=music_track_path,
          has_audio=has_audio,
          music_overlay=video_variant.render_settings.use_music_overlay,
          continuous_audio=video_variant.render_settings.use_continuous_audio,
          full_av_select_filter=full_av_select_filter,
          music_overlay_select_filter=music_overlay_select_filter,
          continuous_audio_select_filter=continuous_audio_select_filter,
      )
    rendered_paths[format_type] = _render_format(
        vision_model=vision_model,
        input_video_path=horizontal_combo_path,
        output_path=output_dir,
        gcs_bucket_name=gcs_bucket_name,
        gcs_folder_path=gcs_folder_path,
        variant_id=video_variant.variant_id,
        format_type=format_type,
        generate_image_assets=(
            video_variant.render_settings.generate_image_assets
        ),
        video_filter=format_instructions['blur_filter'],
        ffmpeg_cmds=format_ffmpeg_cmds,
    )

  StorageService.upload_gcs_dir(
      source_directory=output_dir,
      bucket_name=gcs_bucket_name,
      target_dir=gcs_folder_path,
  )
  result = {'variants': {}}
  if video_variant.render_settings.generate_text_assets:
    text_assets = _generate_text_assets(
        vision_model=vision_model,
        gcs_video_path=(
            f'gs://{gcs_bucket_name}/{gcs_folder_path}/{horizontal_combo_name}'
        ),
        video_language=video_language,
        video_variant=video_variant,
    )
    if text_assets:
      result['texts'] = text_assets

  for format_type, rendered_path in rendered_paths.items():
    result['variants'][format_type] = (
        f'{ConfigService.GCS_BASE_URL}/{gcs_bucket_name}/'
        f'{parse.quote(gcs_folder_path)}/{rendered_path["path"]}'
    )
    if 'images' in rendered_path:
      if 'images' not in result:
        result['images'] = {}
      result['images'][format_type] = rendered_path['images']

  return result


def _get_variant_ffmpeg_commands(
    video_file_path: str,
    speech_track_path: Optional[str],
    music_track_path: Optional[str],
    has_audio: bool,
    music_overlay: bool,
    continuous_audio: bool,
    full_av_select_filter: str,
    music_overlay_select_filter: str,
    continuous_audio_select_filter: str,
):
  ffmpeg_cmds = [
      'ffmpeg',
      '-i',
      video_file_path,
  ]
  if (music_overlay and speech_track_path and music_track_path):
    ffmpeg_cmds.extend([
        '-i',
        speech_track_path,
        '-i',
        music_track_path,
    ])
  ffmpeg_filter = [full_av_select_filter]
  if has_audio:
    if continuous_audio:
      ffmpeg_filter = [continuous_audio_select_filter]
    elif music_overlay:
      ffmpeg_filter = [music_overlay_select_filter, '-ac', '2']
  ffmpeg_cmds.extend([
      '-filter_complex',
  ] + ffmpeg_filter + [
      '-map',
      '[outv]',
  ])
  if has_audio:
    ffmpeg_cmds.extend([
        '-map',
        '[outa]',
    ])

  return ffmpeg_cmds


def _render_format(
    vision_model: GenerativeModel,
    input_video_path: str,
    output_path: str,
    gcs_bucket_name: str,
    gcs_folder_path: str,
    variant_id: int,
    format_type: str,
    generate_image_assets: bool,
    video_filter: str,
    ffmpeg_cmds: Optional[Sequence[str]],
) -> Dict[str, Union[str, Sequence[str]]]:
  """Renders a video variant in a specific format.

  Args:
    vision_model: The generative vision model to use.
    input_video_path: The path to the input video to render.
    output_path: The path to output to.
    gcs_bucket_name: The name of the GCS bucket to upload to.
    gcs_folder_path: The path to the GCS folder to upload to.
    variant_id: The id of the variant to render.
    format_type: The type of the output format (horizontal, vertical, square).
    generate_image_assets: Whether to generate image assets for the variant.
    video_filter: The ffmpeg video filter to use.
    ffmpeg_cmds: The ffmpeg commands to use.
  Returns:
    The rendered video's format name.
  """
  logging.info(
      'RENDERING - Rendering variant %s format: %s', variant_id, format_type
  )
  _, video_ext = os.path.splitext(input_video_path)
  format_name = f'combo_{variant_id}_{format_type[0]}{video_ext}'
  output_video_path = str(pathlib.Path(output_path, format_name))

  if ffmpeg_cmds:
    ffmpeg_cmds.append(output_video_path)
    Utils.execute_subprocess_commands(
        cmds=ffmpeg_cmds,
        description=(
            f'render {format_type} variant with id {variant_id} using ffmpeg'
        ),
    )
  else:
    Utils.execute_subprocess_commands(
        cmds=[
            'ffmpeg',
            '-y',
            '-i',
            input_video_path,
            '-vf',
            video_filter,
            output_video_path,
        ],
        description=(
            f'render {format_type} variant with id {variant_id} and '
            'blur filter using ffmpeg'
        ),
    )
  output = {
      'path': format_name,
  }
  if generate_image_assets:
    StorageService.upload_gcs_dir(
        source_directory=output_path,
        bucket_name=gcs_bucket_name,
        target_dir=gcs_folder_path,
    )
    assets = _generate_image_assets(
        vision_model=vision_model,
        video_file_path=output_video_path,
        gcs_bucket_name=gcs_bucket_name,
        gcs_folder_path=gcs_folder_path,
        output_path=output_path,
        variant_id=variant_id,
        format_type=format_type,
    )
    if assets:
      output['images'] = assets

  return output


def _generate_text_assets(
    vision_model: GenerativeModel,
    gcs_video_path: str,
    video_language: str,
    video_variant: VideoVariant,
) -> Optional[Sequence[Dict[str, str]]]:
  """Generates text ad assets for a video variant.

  Args:
    vision_model: The vision model to use for text generation.
    gcs_video_path: The path to the video to generate text assets for.
    video_language: The language of the video.
    video_variant: The video variant to use for text generation.

  Returns:
    The generated text assets.
  """
  prompt = ConfigService.GENERATE_ASSETS_PROMPT.format(
      video_language=video_language
  )
  assets = None
  try:
    response = vision_model.generate_content(
        [
            Part.from_uri(gcs_video_path, mime_type='video/mp4'),
            prompt,
        ],
        generation_config=ConfigService.GENERATE_ASSETS_CONFIG,
        safety_settings=ConfigService.CONFIG_DEFAULT_SAFETY_CONFIG,
    )
    if (
        response.candidates and response.candidates[0].content.parts
        and response.candidates[0].content.parts[0].text
    ):
      logging.info(
          'ASSETS - Received response: %s',
          response.candidates[0].content.parts[0].text
      )
      rows = []
      results = list(
          filter(
              None, response.candidates[0].content.parts[0].text.strip().split(
                  ConfigService.GENERATE_ASSETS_SEPARATOR
              )
          )
      )
      for result in results:
        result = re.findall(
            ConfigService.GENERATE_ASSETS_PATTERN, result, re.MULTILINE
        )
        rows.append([entry.strip() for entry in result[0]])
      assets = pd.DataFrame(rows, columns=[
          'headline',
          'description',
      ]).to_dict(orient='records')
      logging.info(
          'ASSETS - Generated text assets for variant %d: %r',
          video_variant.variant_id,
          assets,
      )
    else:
      logging.warning(
          'ASSETS - Could not generate text assets for variant %d!',
          video_variant.variant_id
      )
  # Execution should continue regardless of the underlying exception
  # pylint: disable=broad-exception-caught
  except Exception:
    logging.exception(
        'Encountered error during generation of text assets for variant %d! '
        'Continuing...', video_variant.variant_id
    )
  return assets


def _generate_video_script(
    optimised_av_segments: pd.DataFrame,
    video_variant: VideoVariant,
) -> str:
  """Generates a video script for the given A/V segments.

  Args:
    optimised_av_segments: The optimised AV segments to use.
    video_variant: The video variant to use.

  Returns:
    The generated video script.
  """
  video_script = []
  index = 1
  for av_segment in optimised_av_segments:
    if str(av_segment['av_segment_id']) not in video_variant.av_segments.keys():
      continue

    video_script.append(f'Scene {index}')
    video_script.append(f"{av_segment['start_s']} --> {av_segment['end_s']}")
    video_script.append(
        f"Duration: {(av_segment['end_s'] - av_segment['start_s']):.2f}s"
    )

    description = av_segment['description'].strip()
    if description:
      video_script.append(description)

    video_script.append(
        f"Number of visual shots: {len(av_segment['visual_segment_ids'])}"
    )
    transcript = av_segment['transcript']
    details = av_segment['labels'] + av_segment['objects']
    text = [f'"{t}"' for t in av_segment['text']]
    logos = av_segment['logos']
    keywords = av_segment['keywords'].strip()

    if transcript:
      video_script.append(f"Off-screen speech: \"{' '.join(transcript)}\"")
    if details:
      video_script.append(f"On-screen details: {', '.join(details)}")
    if text:
      video_script.append(f"On-screen text: {', '.join(text)}")
    if logos:
      video_script.append(f"Logos: {', '.join(logos)}")
    if keywords:
      video_script.append(f'Keywords: {keywords}')

    video_script.append('')
    index += 1

  video_script = '\n'.join(video_script)
  return video_script


def _generate_image_assets(
    vision_model: GenerativeModel,
    video_file_path: str,
    gcs_bucket_name: str,
    gcs_folder_path: str,
    output_path: str,
    variant_id: int,
    format_type: str,
) -> Sequence[str]:
  """Generates image ad assets for a video variant in a specific format.

  Args:
    vision_model: The generative vision model to use.
    video_file_path: The path to the input video to use.
    gcs_bucket_name: The name of the GCS bucket to upload the assets to.
    gcs_folder_path: The path to the GCS folder to upload the assets to.
    output_path: The path to output to.
    variant_id: The id of the variant to render.
    format_type: The type of the output format (horizontal, vertical, square).

  Returns:
    The paths to the generated image assets.
  """
  variant_folder = f'combo_{variant_id}'
  image_assets_path = pathlib.Path(
      output_path,
      variant_folder,
      ConfigService.OUTPUT_COMBINATION_ASSETS_DIR,
      format_type,
  )
  assets = []
  try:
    os.makedirs(image_assets_path, exist_ok=True)
    _extract_video_thumbnails(
        video_file_path=video_file_path,
        image_assets_path=image_assets_path,
        variant_id=variant_id,
        format_type=format_type,
    )
    _identify_and_extract_key_frames(
        vision_model=vision_model,
        video_file_path=video_file_path,
        image_assets_path=image_assets_path,
        gcs_bucket_name=gcs_bucket_name,
        gcs_folder_path=gcs_folder_path,
        output_path=output_path,
        variant_id=variant_id,
        format_type=format_type,
    )
    assets = [
        f'{ConfigService.GCS_BASE_URL}/{gcs_bucket_name}/'
        f'{parse.quote(gcs_folder_path)}/'
        f'{variant_folder}/{ConfigService.OUTPUT_COMBINATION_ASSETS_DIR}/'
        f'{format_type}/{image_asset}' for image_asset in sorted(
            os.listdir(image_assets_path), key=lambda asset:
            int(asset.split('/')[-1].replace('.png', '').replace('.jpg', ''))
        ) if image_asset.endswith('.png') or image_asset.endswith('.jpg')
    ]

    logging.info(
        'ASSETS - Generated %d image assets for variant %d in %s format',
        len(assets),
        variant_id,
        format_type,
    )
  # Execution should continue regardless of the underlying exception
  # pylint: disable=broad-exception-caught
  except Exception:
    logging.exception(
        'Encountered error during generation of image assets for variant %d '
        'in format %s! Continuing...', variant_id, format_type
    )
  return assets


def _extract_video_thumbnails(
    video_file_path: str,
    image_assets_path: str,
    variant_id: int,
    format_type: str,
):
  """Extracts video thumbnails as image assets for a video in a specific format.

  Args:
    video_file_path: The path to the input video to use.
    image_assets_path: The path to store image assets in.
    variant_id: The id of the variant to render.
    format_type: The type of the output format (horizontal, vertical, square).
  """
  Utils.execute_subprocess_commands(
      cmds=[
          'ffmpeg',
          '-i',
          video_file_path,
          '-vf',
          'thumbnail',
          '-vsync',
          'vfr',
          str(pathlib.Path(image_assets_path, '%d.png')),
      ],
      description=(
          f'extract thumbnails for {format_type} type for '
          f'variant with id {variant_id} using ffmpeg'
      ),
  )


def _identify_and_extract_key_frames(
    vision_model: GenerativeModel,
    video_file_path: str,
    image_assets_path: str,
    gcs_bucket_name: str,
    gcs_folder_path: str,
    output_path: str,
    variant_id: int,
    format_type: str,
):
  """Identifies key frames via Gemini and extracts them from the given video.

  Args:
    vision_model: The generative vision model to use.
    video_file_path: The path to the input video to use.
    image_assets_path: The path to store image assets in.
    gcs_bucket_name: The name of the GCS bucket to upload the assets to.
    gcs_folder_path: The path to the GCS folder to upload the assets to.
    output_path: The path to output to.
    variant_id: The id of the variant to render.
    format_type: The type of the output format (horizontal, vertical, square).
  """
  results = []
  try:
    gcs_video_file_path = video_file_path.replace(f'{output_path}/', '')
    response = vision_model.generate_content(
        [
            Part.from_uri(
                f'gs://{gcs_bucket_name}/{gcs_folder_path}/'
                f'{gcs_video_file_path}',
                mime_type='video/mp4',
            ),
            ConfigService.KEY_FRAMES_PROMPT,
        ],
        generation_config=ConfigService.KEY_FRAMES_CONFIG,
        safety_settings=ConfigService.CONFIG_DEFAULT_SAFETY_CONFIG,
    )
    if (
        response.candidates and response.candidates[0].content.parts
        and response.candidates[0].content.parts[0].text
    ):
      text = response.candidates[0].content.parts[0].text
      results = re.findall(ConfigService.KEY_FRAMES_PATTERN, text, re.MULTILINE)
    else:
      logging.warning('ASSETS - Could not identify key frames!')
  # Execution should continue regardless of the underlying exception
  # pylint: disable=broad-exception-caught
  except Exception:
    logging.exception('Encountered error while identifying key frames!')

  if results:
    for index, key_frame_timestamp in enumerate(results):
      Utils.execute_subprocess_commands(
          cmds=[
              'ffmpeg',
              '-ss',
              key_frame_timestamp,
              '-i',
              video_file_path,
              '-frames:v',
              '1',
              '-q:v',
              '2',
              str(pathlib.Path(image_assets_path, f'{index+1}.jpg')),
          ],
          description=(
              f'extract key frames for {format_type} type for '
              f'variant with id {variant_id} using ffmpeg'
          ),
      )


def _group_consecutive_segments(
    av_segment_ids: Sequence[str],
) -> Sequence[Tuple[str, str]]:
  """Groups consecutive segments together.

  Consecutive A/V segments, such as `1, 2, 3`, will be grouped as a tuple of the
  start and end A/V segment ids, such as `(1, 3)`. This also applies to
  consecutive split segments, for example `4.2, 4.3, 4.4` yields `(4.2, 4.4)`.
  Since we cannot discern whether split segments have ended, we cannot group a
  split segment with a subsequent non-split segment, however we are able to
  group non-split segments with consecutive split-segments only if the
  consecutive split-segments start at `n.1`. For example,
  `1, 2, 3, 4.1, 4.2, 4.3, 5` yields `[(1, 4.3), (5, 5)]` while
  `1, 2, 3, 4.2, 4.3, 4.4, 5` yields `[(1, 3), (4.2, 4.4), (5, 5)]`. Segments
  that are unordered are also treated as individual non-sequential segments.
  For example, `5, 1, 2, 3` yields `[(5, 5), (1, 3)]`.

  Args:
    av_segment_ids: The A/V segments ids to be grouped.

  Returns:
    A sequence of tuples, where each tuple contains the start and end A/V
    segment ids of a group.
  """
  result = []
  i = 0
  while i < len(av_segment_ids):
    start_segment = av_segment_ids[i]
    j = i
    while j < len(av_segment_ids) - 1:
      if _is_sequential_segments(av_segment_ids[j], av_segment_ids[j + 1]):
        j += 1
      else:
        break
    result.append((start_segment, av_segment_ids[j]))
    i = j + 1
  return result


def _is_sequential_segments(current_segment_id: str, next_segment_id: str):
  """Checks if two consecutive segments are sequential.

  Rules:
  1. Incrementing the last numerical part (e.g., '1' -> '2', '4.2' -> '4.3',
     '4.2.1' -> '4.2.2').
  2. Stepping down to a new sub-level by appending '.1' to the segment ID
      that would be the result of incrementing the last part of the current ID
      (e.g., '3' -> '4.1' because incrementing '3' gives '4', and '4.1' is
      '4' + '.1';
      '4.1' -> '4.2.1' because incrementing '4.1' gives '4.2', and '4.2.1' is
      '4.2' + '.1').

  Args:
    current_segment_id: The current segment ID.
    next_segment_id: The next segment ID.

  Returns:
    True if the next segment is sequential to the current, False otherwise.
  """
  current_parts = current_segment_id.split('.')
  next_parts = next_segment_id.split('.')
  try:
    next_level_parts = list(current_parts)
    next_level_parts[-1] = str(int(next_level_parts[-1]) + 1)
    if (
        len(current_parts) == len(next_parts)
        and current_parts[:-1] == next_parts[:-1]
        and int(next_parts[-1]) == int(current_parts[-1]) + 1
    ) or next_segment_id == '.'.join(next_level_parts) + '.1':
      return True
  except ValueError:
    pass
  return False


def _build_ffmpeg_filters(
    shot_timestamps: Sequence[Tuple[float, float]],
    has_audio: bool,
    render_settings: VideoVariantRenderSettings,
    video_duration: float,
) -> Tuple[str, str, str]:
  """Builds the ffmpeg filters.

  Args:
    shot_timestamps: A sequence of tuples, where each tuple contains the start
      and end timestamps of a shot.
    has_audio: Whether the video has audio.
    render_settings: The render settings to use.
    video_duration: The duration of the video.

  Returns:
    A tuple containing the full audio/video, music overlay and continuous audio
    ffmpeg filters.
  """
  video_select_filter = []
  audio_select_filter = []
  select_filter_concat = []
  idx = 0
  duration = 0
  variant_first_segment_start = sys.maxsize
  variant_last_segment_end = 0
  for start, end in shot_timestamps:
    selection_filter = f'between(t,{start},{end})'
    video_select_filter.append(
        f"[0:v]select='{selection_filter}',setpts=N/FRAME_RATE/TB[v{idx}];"
    )
    select_filter_concat.append(f'[v{idx}]')
    if has_audio:
      audio_select_filter.append(
          f"[0:a]aselect='{selection_filter}',asetpts=N/SR/TB[a{idx}];"
      )
      select_filter_concat.append(f'[a{idx}]')
    duration += end - start
    variant_first_segment_start = min(variant_first_segment_start, start)
    variant_last_segment_end = max(variant_last_segment_end, end)
    idx += 1

  fade_out_duration = float(ConfigService.CONFIG_DEFAULT_FADE_OUT_DURATION)
  fade_out_buffer = float(ConfigService.CONFIG_DEFAULT_FADE_OUT_BUFFER)
  fade_out_start = duration - fade_out_duration - fade_out_buffer
  fade_out_filter = (
      f';[outa]afade=t=out:st={fade_out_start}:d={fade_out_duration}[outa]'
      if render_settings.fade_out else ''
  )

  match render_settings.overlay_type:
    case Utils.RenderOverlayType.VIDEO_START.value:
      overlay_start = 0
    case Utils.RenderOverlayType.VIDEO_END.value:
      overlay_start = video_duration - duration
    case Utils.RenderOverlayType.VARIANT_END.value:
      overlay_start = variant_last_segment_end - duration
    case Utils.RenderOverlayType.VARIANT_START.value | _:
      overlay_start = variant_first_segment_start

  full_av_select_filter = ''.join(
      video_select_filter + audio_select_filter + select_filter_concat
      + [f'concat=n={idx}:v=1:a=1[outv][outa]', fade_out_filter]
  ) if has_audio else ''.join(
      video_select_filter + select_filter_concat
      + [f'concat=n={idx}:v=1[outv]']
  )

  music_overlay_select_filter = ''.join(
      video_select_filter
      + [entry.replace('0:a', '1:a') for entry in audio_select_filter] + [
          f"[2:a]aselect='between(t,{overlay_start},{overlay_start+duration})'"
          ',asetpts=N/SR/TB[music];'
      ] + select_filter_concat + [
          f'concat=n={idx}:v=1:a=1[outv][tempa];',
          '[tempa][music]amerge=inputs=2[outa]',
          fade_out_filter,
      ]
  ) if has_audio else ''
  continuous_audio_select_filter = ''.join(
      video_select_filter + [
          f"[0:a]aselect='between(t,{overlay_start},{overlay_start+duration})'"
          ',asetpts=N/SR/TB[outa];'
      ] + [entry for entry in select_filter_concat if entry.startswith('[v')]
      + [f'concat=n={idx}:v=1[outv]', fade_out_filter]
  ) if has_audio else ''

  return (
      full_av_select_filter,
      music_overlay_select_filter,
      continuous_audio_select_filter,
  )


import json
import logging
import os
import pathlib
import re
import subprocess
import tempfile
import time
from typing import Any, Dict, Sequence, Tuple


class EndSlateProcessor:
    """Processor for adding end slate images to rendered videos.

    Creates a complete copy of the render folder with end slate videos.
    All files are copied to ensure independence from original folder.
    """

    def __init__(self, gcs_bucket_name: str, trigger_file: Utils.TriggerFile):
        """Initializes the EndSlateProcessor."""
        self.gcs_bucket_name = gcs_bucket_name
        self.trigger_file = trigger_file
        self.tmp_dir = tempfile.mkdtemp()
        logging.info('END_SLATE - Initialized with temp dir: %s', self.tmp_dir)

    def process(self):
        """Main processing function - Creates complete CTA folder."""
        logging.info('END_SLATE - Starting end slate processing...')

        try:
            # Step 1: Load config
            config = self._load_config()

            # Step 2: Download end slate image
            image_path = self._download_image(config['image'])

            # Step 3: Get image dimensions
            image_width, image_height = self._get_image_dimensions(image_path)
            logging.info(
                'END_SLATE - Image dimensions: %dx%d',
                image_width,
                image_height
            )

            # Step 4: Create CTA folder name
            render_folder = config['rendered_video_folder']
            timestamp = int(time.time() * 1000)
            folder_name_base = render_folder.replace('--combos', '').split('--')[0]
            cta_folder = f'CTA - {folder_name_base}--{timestamp}-combos'

            logging.info('END_SLATE - Creating CTA folder: %s', cta_folder)

            # Step 5: Get ALL files from original folder
            all_files = self._get_all_files_in_folder(render_folder)
            logging.info(
                'END_SLATE - Found %d total files to process',
                len(all_files)
            )

            # Step 6: Categorize files
            video_files = [f for f in all_files if self._is_combo_video(f)]
            text_files = [f for f in all_files if self._is_text_file(f)]
            other_files = [f for f in all_files
                           if f not in video_files and f not in text_files]

            logging.info(
                'END_SLATE - Categorized: %d videos, %d text files, %d other files',
                len(video_files),
                len(text_files),
                len(other_files)
            )

            # Step 7: Process video files (add end slate)
            duration = config['duration']
            for video_file in video_files:
                self._process_video_with_endslate(
                    video_gcs_path=video_file,
                    image_path=image_path,
                    duration=duration,
                    original_folder=render_folder,
                    cta_folder=cta_folder
                )

            # Step 8: Copy and update text files
            for text_file in text_files:
                self._copy_and_update_text_file(
                    text_file_gcs_path=text_file,
                    original_folder=render_folder,
                    cta_folder=cta_folder
                )

            # Step 9: Copy all other files as-is
            for other_file in other_files:
                self._copy_file_as_is(
                    file_gcs_path=other_file,
                    original_folder=render_folder,
                    cta_folder=cta_folder
                )

            logging.info('END_SLATE - Processing completed successfully!')
            logging.info('END_SLATE - CTA folder created: %s', cta_folder)

        except Exception as e:
            logging.exception('END_SLATE - Error during processing: %s', str(e))
            raise

    def _load_config(self) -> Dict[str, Any]:
        """Loads and parses the end slate configuration JSON."""
        from urllib import parse

        logging.info(
            'END_SLATE - Loading config from: %s',
            self.trigger_file.full_gcs_path
        )

        json_content = StorageService.download_gcs_file(
            file_path=self.trigger_file,
            bucket_name=self.gcs_bucket_name,
            fetch_contents=True
        )

        if not json_content:
            raise ValueError('Failed to download end slate config JSON')

        config = json.loads(json_content)

        # Validate required fields
        required_fields = ['image', 'duration', 'rendered_video_folder']
        for field in required_fields:
            if field not in config:
                raise ValueError(f'Missing required field in config: {field}')

        # URL decode folder name
        config['rendered_video_folder'] = parse.unquote(
            config['rendered_video_folder']
        )

        # Convert duration to int
        config['duration'] = int(config['duration'])

        logging.info('END_SLATE - Config loaded: %s', config)
        return config

    def _download_image(self, image_filename: str) -> str:
        """Downloads the end slate image from GCS."""
        image_gcs_path = str(
            pathlib.Path(self.trigger_file.gcs_root_folder, image_filename)
        )

        logging.info(
            'END_SLATE - Downloading image from: gs://%s/%s',
            self.gcs_bucket_name,
            image_gcs_path
        )

        image_trigger = Utils.TriggerFile(image_gcs_path)

        local_path = StorageService.download_gcs_file(
            file_path=image_trigger,
            bucket_name=self.gcs_bucket_name,
            output_dir=self.tmp_dir
        )

        if not local_path:
            raise ValueError(
                f'Failed to download image from: '
                f'gs://{self.gcs_bucket_name}/{image_gcs_path}'
            )

        logging.info('END_SLATE - Image downloaded to: %s', local_path)
        return local_path

    def _get_all_files_in_folder(self, render_folder: str):
        """Gets ALL files in the render folder (recursive)."""
        full_folder_path = str(
            pathlib.Path(
                self.trigger_file.gcs_root_folder,
                render_folder
            )
        )

        # Ensure trailing slash for prefix matching
        if not full_folder_path.endswith('/'):
            full_folder_path += '/'

        logging.info('END_SLATE - Scanning folder: %s', full_folder_path)

        # Use Google Cloud Storage directly (most reliable)
        from google.cloud import storage

        storage_client = storage.Client()
        bucket = storage_client.bucket(self.gcs_bucket_name)

        # List all files with this prefix
        blobs = bucket.list_blobs(prefix=full_folder_path)

        # Convert to list of paths, excluding folder markers
        all_files = [blob.name for blob in blobs if not blob.name.endswith('/')]

        logging.info('END_SLATE - Found %d files', len(all_files))

        return all_files

    def _is_combo_video(self, file_path: str) -> bool:
        """Checks if file is a combo video that needs end slate."""
        pattern = r'combo_\d+_[hsv]\.mp4$'
        return bool(re.search(pattern, file_path))

    def _is_text_file(self, file_path: str) -> bool:
        """Checks if file is a text file that might contain URLs."""
        text_extensions = ['.json', '.txt', '.yaml', '.yml', '.xml', '.html']
        return any(file_path.lower().endswith(ext) for ext in text_extensions)

    def _get_relative_path(self, file_gcs_path: str, folder_name: str) -> str:
        """Gets the relative path of a file within its folder."""
        # Extract path after folder name
        parts = file_gcs_path.split(folder_name)
        if len(parts) > 1:
            relative = parts[1].lstrip('/')
            return relative
        return os.path.basename(file_gcs_path)

    def _get_image_dimensions(self, image_path: str) -> Tuple[int, int]:
        """Gets image dimensions using ffprobe."""
        cmd = [
            'ffprobe',
            '-v', 'quiet',
            '-print_format', 'json',
            '-show_streams',
            image_path
        ]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True
        )

        data = json.loads(result.stdout)

        video_stream = next(
            (s for s in data['streams'] if s['codec_type'] == 'video'),
            None
        )

        if not video_stream:
            raise ValueError(f'Could not detect dimensions for image: {image_path}')

        width = int(video_stream['width'])
        height = int(video_stream['height'])

        return width, height

    def _get_video_info(self, video_path: str) -> Dict[str, Any]:
        """Gets video metadata using ffprobe."""
        cmd = [
            'ffprobe',
            '-v', 'quiet',
            '-print_format', 'json',
            '-show_streams',
            '-select_streams', 'v:0',
            video_path
        ]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True
        )

        data = json.loads(result.stdout)
        video_stream = data['streams'][0]

        # Parse frame rate
        fps_str = video_stream.get('r_frame_rate', '30/1')
        if '/' in fps_str:
            num, denom = map(int, fps_str.split('/'))
            fps = num / denom
        else:
            fps = float(fps_str)

        return {
            'width': int(video_stream['width']),
            'height': int(video_stream['height']),
            'fps': fps,
            'codec': video_stream['codec_name']
        }

    def _process_video_with_endslate(
            self,
            video_gcs_path: str,
            image_path: str,
            duration: int,
            original_folder: str,
            cta_folder: str
    ):
        """Processes a video file by adding end slate."""
        relative_path = self._get_relative_path(video_gcs_path, original_folder)
        logging.info('END_SLATE - Processing video: %s', relative_path)

        # Download video
        video_trigger = Utils.TriggerFile(video_gcs_path)
        local_video_path = StorageService.download_gcs_file(
            file_path=video_trigger,
            bucket_name=self.gcs_bucket_name,
            output_dir=self.tmp_dir
        )

        if not local_video_path:
            logging.error(
                'END_SLATE - Failed to download video: %s',
                video_gcs_path
            )
            return

        # Get video info
        video_info = self._get_video_info(local_video_path)
        logging.info(
            'END_SLATE - Video info: %dx%d @ %.2f fps',
            video_info['width'],
            video_info['height'],
            video_info['fps']
        )

        # Create output path
        output_filename = f'endslate_{os.path.basename(local_video_path)}'
        output_path = str(pathlib.Path(self.tmp_dir, output_filename))

        # Add end slate using FFmpeg
        self._add_end_slate_ffmpeg(
            input_video=local_video_path,
            image_path=image_path,
            output_path=output_path,
            duration=duration,
            video_width=video_info['width'],
            video_height=video_info['height'],
            fps=video_info['fps']
        )

        # Upload to CTA folder with same relative path
        new_video_gcs_path = str(pathlib.Path(
            self.trigger_file.gcs_root_folder,
            cta_folder,
            relative_path
        ))

        logging.info(
            'END_SLATE - Uploading processed video to: %s',
            new_video_gcs_path
        )

        StorageService.upload_gcs_file(
            file_path=output_path,
            destination_file_name=new_video_gcs_path,
            bucket_name=self.gcs_bucket_name,
            overwrite=False
        )

        logging.info('END_SLATE - Successfully processed: %s', relative_path)

    def _add_end_slate_ffmpeg(
            self,
            input_video: str,
            image_path: str,
            output_path: str,
            duration: int,
            video_width: int,
            video_height: int,
            fps: float
    ):
        """Adds end slate to video using FFmpeg."""
        # Fix SAR mismatch by explicitly setting SAR to 1:1
        filter_complex = (
            f'[1:v]scale={video_width}:{video_height}:'
            f'force_original_aspect_ratio=decrease,'
            f'pad={video_width}:{video_height}:(ow-iw)/2:(oh-ih)/2:color=black,'
            f'setsar=1:1[img];'
            f'[0:v]setsar=1:1[v];'
            f'[v][img]concat=n=2:v=1:a=0[outv]'
        )

        cmd = [
            'ffmpeg',
            '-y',
            '-i', input_video,
            '-loop', '1',
            '-framerate', str(fps),
            '-t', str(duration),
            '-i', image_path,
            '-filter_complex', filter_complex,
            '-map', '[outv]',
            '-map', '0:a?',
            '-c:v', 'libx264',
            '-preset', 'fast',
            '-crf', '23',
            '-c:a', 'copy',
            '-movflags', '+faststart',
            output_path
        ]

        logging.info('END_SLATE - Running FFmpeg command: %s', ' '.join(cmd))

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True
        )

        if result.returncode != 0:
            logging.error('END_SLATE - FFmpeg error: %s', result.stderr)
            raise RuntimeError(f'FFmpeg failed with error: {result.stderr}')

        logging.info('END_SLATE - FFmpeg completed successfully')

    def _copy_and_update_text_file(
            self,
            text_file_gcs_path: str,
            original_folder: str,
            cta_folder: str
    ):
        """Copies a text file and updates folder references."""
        relative_path = self._get_relative_path(text_file_gcs_path, original_folder)
        logging.info('END_SLATE - Processing text file: %s', relative_path)

        # Download text file
        text_trigger = Utils.TriggerFile(text_file_gcs_path)
        content = StorageService.download_gcs_file(
            file_path=text_trigger,
            bucket_name=self.gcs_bucket_name,
            fetch_contents=True
        )

        if not content:
            logging.error(
                'END_SLATE - Failed to download text file: %s',
                text_file_gcs_path
            )
            return

        # ✅ FIX: Convert bytes to string if needed
        if isinstance(content, bytes):
            try:
                content = content.decode('utf-8')
            except UnicodeDecodeError:
                # Try with latin-1 as fallback
                content = content.decode('latin-1')

        # Update content: replace all references to original folder
        updated_content = self._update_folder_references(
            content=content,
            original_folder=original_folder,
            cta_folder=cta_folder
        )

        # Special handling for combos.json
        if text_file_gcs_path.endswith('combos.json'):
            updated_content = self._update_combos_json(
                content=updated_content,
                cta_folder=cta_folder
            )

        # Save locally
        local_path = str(pathlib.Path(self.tmp_dir, f'updated_{relative_path.replace("/", "_")}'))
        with open(local_path, 'w', encoding='utf-8') as f:
            f.write(updated_content)

        # Upload to CTA folder with same relative path
        new_text_gcs_path = str(pathlib.Path(
            self.trigger_file.gcs_root_folder,
            cta_folder,
            relative_path
        ))

        logging.info(
            'END_SLATE - Uploading updated text file to: %s',
            new_text_gcs_path
        )

        StorageService.upload_gcs_file(
            file_path=local_path,
            destination_file_name=new_text_gcs_path,
            bucket_name=self.gcs_bucket_name,
            overwrite=True
        )

        logging.info('END_SLATE - Successfully updated text file: %s', relative_path)

    def _update_folder_references(
            self,
            content: str,
            original_folder: str,
            cta_folder: str
    ) -> str:
        """Updates all references to original folder with CTA folder."""
        # URL-encoded version
        import urllib.parse
        original_encoded = urllib.parse.quote(original_folder, safe='')
        cta_encoded = urllib.parse.quote(cta_folder, safe='')

        # Replace both plain and encoded versions
        updated = content.replace(original_folder, cta_folder)
        updated = updated.replace(original_encoded, cta_encoded)

        return updated

    def _update_combos_json(self, content: str, cta_folder: str) -> str:
        """Special updates for combos.json."""
        try:
            data = json.loads(content)

            # Update each variant
            for key, variant in data.items():
                if not key.startswith('_'):
                    continue

                # Update title
                if 'title' in variant:
                    if 'WITH END SLATE' not in variant['title']:
                        variant['title'] = variant['title'] + ' - WITH END SLATE'

                # Update description
                if 'description' in variant:
                    if 'एंड स्लेट के साथ' not in variant['description']:
                        variant['description'] = variant['description'] + ' एंड स्लेट के साथ।'

            return json.dumps(data, indent=2, ensure_ascii=False)

        except json.JSONDecodeError:
            logging.warning('END_SLATE - Failed to parse combos.json, returning as-is')
            return content

    def _copy_file_as_is(
            self,
            file_gcs_path: str,
            original_folder: str,
            cta_folder: str
    ):
        """Copies a file as-is without modifications."""
        relative_path = self._get_relative_path(file_gcs_path, original_folder)
        logging.info('END_SLATE - Copying file: %s', relative_path)

        # Download file
        file_trigger = Utils.TriggerFile(file_gcs_path)
        local_path = StorageService.download_gcs_file(
            file_path=file_trigger,
            bucket_name=self.gcs_bucket_name,
            output_dir=self.tmp_dir
        )

        if not local_path:
            logging.error(
                'END_SLATE - Failed to download file: %s',
                file_gcs_path
            )
            return

        # Upload to CTA folder with same relative path
        new_file_gcs_path = str(pathlib.Path(
            self.trigger_file.gcs_root_folder,
            cta_folder,
            relative_path
        ))

        logging.info(
            'END_SLATE - Uploading copied file to: %s',
            new_file_gcs_path
        )

        StorageService.upload_gcs_file(
            file_path=local_path,
            destination_file_name=new_file_gcs_path,
            bucket_name=self.gcs_bucket_name,
            overwrite=True
        )

        logging.info('END_SLATE - Successfully copied: %s', relative_path)

