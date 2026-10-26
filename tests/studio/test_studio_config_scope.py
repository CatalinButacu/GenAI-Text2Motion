import inspect

import yaml

from text2motion.studio.avatar import SmplxAvatarController

LIVE_FIT_KEYS = {"model_dir", "device", "threads", "gender", "num_betas"}
LIVE_SCHEDULE_KEYS = {"retrofit_window"}


def test_the_studio_file_holds_only_settings_that_the_live_studio_reads():
    with open("configs/studio.yml", encoding="utf-8") as handle:
        studio = yaml.safe_load(handle)

    assert set(studio["fit"]) <= LIVE_FIT_KEYS
    assert set(studio["schedule"]) <= LIVE_SCHEDULE_KEYS


def test_the_live_fit_takes_no_warm_start():
    parameters = inspect.signature(SmplxAvatarController.fit_single_chunk).parameters

    assert "warm_start" not in parameters
