"""Reject misleading B-size comparisons before statistical reporting."""
import hashlib
import json

import numpy as np
import pytest

from ipad_jepa.backbone_audit import check_small_cache, check_phase_selection, check_bank, check_pins


def cache_fixture(split="calibration"):
    row = {"device":"R01", "partition":"training", "sequence":"01", "split":split,
           "frames":32, "names_sha256":"names", "frames_content_sha256":"images"}
    smoke = {"model":"dinov3-b", "weight_sha256":"weights", "upstream_commit":"upstream",
             "adapter_sha256":"adapter", "reader_sha256":"base-reader"}
    ids = np.arange(0,32,4,dtype=np.int64) if split == "fit" else np.arange(15,32,dtype=np.int64)
    spec = {"backbone":"dinov3-b", "mode":"online", "feature_dimension":768, "clip_frames":16,
            "fit_stride":4, "image_size":384, "device":"R01", "partition":"training", "sequence":"01",
            "split":split, "frames":32, "padding":split == "fit", "frame_names_sha256":"names",
            "frame_content_sha256":"images", "weights_sha256":"weights", "upstream_commit":"upstream",
            "adapter_sha256":"adapter", "base_reader_sha256":"base-reader", "rows":len(ids),
            "preprocessing":"RGB full-frame PIL bilinear resize; ImageNet mean/std",
            "feature_dtype":"float16 from BF16 inference"}
    meta = {"status":"complete", "spec":spec}
    rehash(meta)
    return meta,row,smoke,ids,np.zeros((len(ids),576,768),np.float16),np.zeros((len(ids),768),np.float16)


def rehash(meta):
    meta["fingerprint"] = hashlib.sha256(json.dumps(meta["spec"],sort_keys=True).encode()).hexdigest()


def test_b_cache_accepts_normal_fit_left_padding_and_complete_calibration():
    for split in ["fit","calibration"]:check_small_cache(*cache_fixture(split))


@pytest.mark.parametrize("field,value", [("mode","offline"),("feature_dimension",1024),("clip_frames",8),
                                           ("padding",True),("weights_sha256","other")])
def test_b_cache_rejects_semantic_changes_even_with_rehashed_metadata(field,value):
    args = cache_fixture(); args[0]["spec"][field] = value; rehash(args[0])
    with pytest.raises(ValueError,match="identity"):check_small_cache(*args)


def test_b_cache_rejects_future_or_missing_target_and_wrong_width():
    args = list(cache_fixture()); args[3] = args[3].copy(); args[3][-1] += 1
    with pytest.raises(AssertionError):check_small_cache(*args)
    args = list(cache_fixture()); args[4] = np.zeros((17,576,1024),np.float16)
    with pytest.raises(ValueError,match="geometry"):check_small_cache(*args)


def test_b_cache_rejects_test_partition_in_normal_fit():
    args = list(cache_fixture("fit")); args[1]["partition"] = "testing"; args[0]["spec"]["partition"] = "testing"; rehash(args[0])
    with pytest.raises(ValueError,match="normal split"):check_small_cache(*args)


def test_b_cache_rejects_nonfinite_context():
    args = cache_fixture(); args[-1][0,767] = np.nan
    with pytest.raises(ValueError,match="Nonfinite"):check_small_cache(*args)


def test_b_selected_head_uses_normal_ce_and_requires_full_training():
    curve = [{"epoch":i+1,"train_ce":1.,"normal_calibration_ce":2.+abs(i-5),"normal_calibration_circular_mae":.1} for i in range(20)]
    phase = {"status":"complete","epochs":20,"feature_dimension":768,"phase_batch_size":256,
             "selected_epoch":6,"best_normal_calibration_ce":2.,"seed":0}
    check_phase_selection(phase,curve,{"seed":0})
    with pytest.raises(ValueError,match="minimize"):check_phase_selection({**phase,"selected_epoch":1},curve,{"seed":0})
    with pytest.raises(ValueError,match="Twenty"):check_phase_selection(phase,curve[:-1],{"seed":0})
    curve[3]["normal_calibration_ce"] = float("nan")
    with pytest.raises(ValueError,match="Nonfinite"):check_phase_selection(phase,curve,{"seed":0})


def test_b_bank_rejects_wrong_width_and_mutated_normal_temperature(tmp_path):
    values = {"mean":np.zeros(768,np.float32),"components":np.eye(256,768,dtype=np.float32),
              "prototypes":np.zeros((16,128,256),np.float32),"temperature":np.array(.2),"cycle_length":np.array(160.)}
    values["prototypes"][:,:,0] = 1
    path = tmp_path/"bank.npz"; np.savez(path,**values)
    normal = {"temperature":.2,"cycle_length_fit_median":160.}
    with np.load(path,allow_pickle=False) as bank:
        check_bank(bank,normal)
        with pytest.raises(AssertionError):check_bank(bank,{**normal,"temperature":.3})
    values["components"] = np.eye(256,1024,dtype=np.float32); np.savez(path,**values)
    with np.load(path,allow_pickle=False) as bank:
        with pytest.raises(ValueError,match="geometry"):check_bank(bank,normal)


def test_source_pins_are_portable_but_reject_missing_duplicate_and_mutation():
    required = {"scripts/run_small_matrix.py","src/model.py"}
    hashes = {"scripts/run_small_matrix.py":"script","src/model.py":"model"}
    pinned = {"/old/checkout/scripts/run_small_matrix.py":"script","src/model.py":"model"}
    assert check_pins(pinned,required,hashes.__getitem__) == hashes
    with pytest.raises(ValueError,match="missing"):check_pins({"src/model.py":"model"},required,hashes.__getitem__)
    with pytest.raises(ValueError,match="duplicated"):check_pins({**pinned,"scripts/run_small_matrix.py":"script"},required,hashes.__getitem__)
    with pytest.raises(ValueError,match="changed"):check_pins({**pinned,"src/model.py":"mutated"},required,hashes.__getitem__)
