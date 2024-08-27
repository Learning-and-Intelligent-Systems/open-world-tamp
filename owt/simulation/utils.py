import colorsys
import random

import numpy as np
from scipy.interpolate import interp1d

import owt.pb_utils as pbu

X_AXIS = np.array([1, 0, 0])  # TODO: make immutable
Z_AXIS = np.array([0, 0, 1])


COLORS = {"yellow": YELLOW, "green": GREEN}
COLORS.update(CHROMATIC_COLORS)


def wait_until_finish(message="Press enter to finish"):
    wait_if_gui(message=message)
    disconnect()


def compose_fns(fn, *fns):
    # Negate
    def new_fn(*args, **kwargs):
        x = fn(*args, **kwargs)
        for fn2 in fns:
            x = fn2(x)
        return x

    return new_fn


def interpolate_exterior(vertices, step_size=1e-3):
    # Sample points in the interior of the structure for OOBB
    interp_distances = [0.0]
    interp_points = [vertices[0]]
    for v1, v2 in get_wrapped_pairs(vertices):
        interp_distances.append(interp_distances[-1] + get_distance(v1, v2))
        interp_points.append(v2)
    f = interp1d(interp_distances, np.array(interp_points).T, kind="linear")
    return list(map(f, np.arange(interp_distances[0], interp_distances[-1], step_size)))


def sample_convex(points):
    points = list(points)
    while True:
        weights = np.random.random(
            size=len(points)
        )  # TODO: sample a different distribution
        index = random.randint(0, len(points) - 1)
        remaining = 1 - sum(weights) + weights[index]
        if remaining >= 0:
            weights[index] = remaining
            return np.average(points, weights=weights, axis=0)


def sample_norm(mu, sigma, lower=0.0, upper=INF):
    # scipy.stats.truncnorm
    assert lower <= upper
    if lower == upper:
        return lower
    if sigma == 0.0:
        assert lower <= mu <= upper
        return mu
    while True:
        x = random.gauss(mu=mu, sigma=sigma)
        if lower <= x <= upper:
            return x


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


def sorted_union(*collections):
    # TODO: stable versus sorted
    # TODO: OrderedDict
    return sorted(set.union(*map(set, collections)))


def sorted_intersection(*collections):
    # TODO: OrderedDict
    return sorted(set.intersection(*map(set, collections)))


def select_indices(sequence, indices):
    return [sequence[index] for index in indices]


def partition(test, sequence):
    success, failure = [], []
    for element in sequence:
        (success if test(element) else failure).append(element)
    return success, failure


def get_rigid_ancestor(body, target_link, **kwargs):
    # dump_body(body)
    for link in reversed(
        get_link_ancestors(body, target_link, **kwargs) + [target_link]
    ):
        # data = get_visual_data(body, link)
        data = get_collision_data(body, link, **kwargs)
        if data:
            return link
        if (link == BASE_LINK) or is_movable(
            body, parent_joint_from_link(link), **kwargs
        ):
            break
    return None


#######################################################


def get_hue_distance(rgb1, rgb2):
    hsv1 = colorsys.rgb_to_hsv(*remove_alpha(rgb1))
    hsv2 = colorsys.rgb_to_hsv(*remove_alpha(rgb2))
    return interval_distance(hsv1[0], hsv2[0], interval=(0, 1))


def get_color_distance(rgb1, rgb2, hue_only=False):
    # TODO: threshold based on how close to grey, white, black, etc.
    distance_fn = get_hue_distance if hue_only else get_distance
    return distance_fn(remove_alpha(rgb1), remove_alpha(rgb2))


def find_closest_color(color, color_from_name=COLORS, **kwargs):
    # TODO: use the hue instead
    # from colorsys import rgb_to_hsv
    if color is None:
        return color

    closest = min(
        color_from_name,
        key=lambda n: get_color_distance(color_from_name[n], color, **kwargs),
    )
    return closest


def get_matplotlib_colors():
    import matplotlib.colors as mcolors

    color_from_name = {}
    for d in [
        mcolors.BASE_COLORS,
        mcolors.TABLEAU_COLORS,
        mcolors.CSS4_COLORS,
    ]:  # mcolors.XKCD_COLORS
        for name, hex in d.items():
            color_from_name[name] = mcolors.rgb_to_hsv(mcolors.to_rgb(hex))
    return color_from_name


def mean_hue(rgbs, min_sat=0.0, min_value=0.0):
    hsvs = [colorsys.rgb_to_hsv(*remove_alpha(rgb)) for rgb in rgbs]
    hues = [h for h, s, v in hsvs if (s >= min_sat) and (v >= min_value)]
    if not hues:
        return None
    # TODO: circular standard deviation
    from scipy.stats import circmean  # , circstd, circvar

    return circmean(hues, low=0.0, high=1.0)


def random_color(saturation=1.0, value=1.0):
    return colorsys.hsv_to_rgb(h=random.random(), s=saturation, v=value)
