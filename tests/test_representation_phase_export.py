"""Selection guards prevent a partial run or an MAE-selected head being published."""
import pytest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from export_representation_phase import check_curve


def curve():
    rows = [dict(epoch=str(e), train_ce='1.0', normal_calibration_ce=str(2 + abs(e - 7) / 10),
                 normal_calibration_circular_mae='0.02' if e == 19 else '0.04') for e in range(1, 21)]
    info = dict(status='complete', epochs=20, selected_epoch=7, best_normal_calibration_ce=2.)
    return info, rows


def test_ce_selection_does_not_follow_later_mae_minimum():
    info, rows = curve()
    assert check_curve(info, rows)['epoch'] == '7'
    info['selected_epoch'] = 19
    with pytest.raises(ValueError, match='minimum normal CE'):
        check_curve(info, rows)


def test_tied_ce_must_preserve_first_epoch():
    info, rows = curve()
    rows[7]['normal_calibration_ce'] = '2.0'
    info['selected_epoch'] = 8
    with pytest.raises(ValueError, match='first minimum'):
        check_curve(info, rows)


def test_partial_curve_cannot_look_complete():
    info, rows = curve()
    with pytest.raises(ValueError, match='twenty-epoch'):
        check_curve(info, rows[:-1])


@pytest.mark.parametrize('field,value', [('train_ce', 'nan'), ('normal_calibration_circular_mae', '0.6')])
def test_invalid_phase_statistics_are_not_exportable(field, value):
    info, rows = curve()
    rows[0][field] = value
    with pytest.raises(ValueError, match='Finite nonnegative'):
        check_curve(info, rows)
