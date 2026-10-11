import shutil
import subprocess
from types import SimpleNamespace

import pytest

from evaluation.lite.normalization import _probe
from evaluation.lite.suite import _normalize_candidate


@pytest.mark.skipif(not shutil.which('ffmpeg') or not shutil.which('ffprobe'), reason='ffmpeg required')
@pytest.mark.parametrize('source_fps', [12, 16, 17, 20, 24, 60])
def test_timestamp_resampling_has_common_fps_without_stretching_time(tmp_path, source_fps):
    source = tmp_path / 'source.mp4'
    subprocess.run(['ffmpeg','-v','error','-f','lavfi','-i',
                    f'testsrc2=size=64x48:rate={source_fps}:duration=2',
                    '-c:v','libx264','-crf','0',str(source)], check=True, capture_output=True)
    original = source.read_bytes()
    dest = _normalize_candidate(
        case=SimpleNamespace(protocol={'candidate_normalization':'lighting_fps12'}),
        candidate=SimpleNamespace(candidate_video=source, model='test',
                                  case_id='SunRisesAndSets/lighting', seed='42'),
        output_root=tmp_path/'out',
    )
    result = _probe(dest, timeout_sec=30)
    assert result == {'width':64,'height':48,'fps':12.0,'frames':24}
    assert source.read_bytes() == original
