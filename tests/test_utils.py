import xspeds.utils as utils
import numpy as np

# TODO use hypothesis (or something) for some of these


def test_pol_to_cart():
    r = np.random.uniform(0.0, 100.0)
    theta = np.random.uniform(0.0, np.pi)
    phi = np.random.uniform(-np.pi, np.pi)

    # assert converting to cartesian and back is correct to within 1e-6
    assert np.linalg.norm(np.array(utils.cart_to_sph(
        utils.sph_to_cart([r, theta, phi]))) - np.array([r, theta, phi])) < r * 1e-6
    # assert the same thing works with a numpy array
    assert np.linalg.norm(np.array(utils.cart_to_sph(
        utils.sph_to_cart(np.array([r, theta, phi])))) - np.array([r, theta, phi])) < r * 1e-6

def test_D():
    # test 1->1 function
    f = lambda x: np.sin(x)
    df = lambda x: np.cos(x)

    x0 = np.random.uniform(0.0, 2 * np.pi)
    D_res = utils.D(f, 0, args=[x0])

    assert D_res - df(x0) < 1e-9

    x1, y1 = np.random.uniform(0.0, 2 * np.pi), np.random.uniform(0.0, 2 * np.pi)
    # test 2->2 function
    f1 = lambda x, y: [np.cos(x + y), np.sin(x - y)]
    df1_x = lambda x, y: np.array([-np.sin(x + y), np.cos(x - y)])
    df1_y = lambda x, y: np.array([-np.sin(x + y), -np.cos(x - y)])

    D_res1_x = utils.D(f1, 0, args=[x1,y1])
    assert np.linalg.norm(D_res1_x - df1_x(x1,y1)) < 1e-9

    D_res1_y = utils.D(f1, 1, args=[x1,y1])
    assert np.linalg.norm(D_res1_y - df1_y(x1,y1)) < 1e-9

def test_J():
    x0, y0 = np.random.uniform(0.0, 2 * np.pi), np.random.uniform(0.0, 2 * np.pi)

    # test 2->2 function
    f1 = lambda x, y: [np.cos(x + y), np.sin(x - y)]
    df1_x = lambda x, y: np.array([-np.sin(x + y), np.cos(x - y)])
    df1_y = lambda x, y: np.array([-np.sin(x + y), -np.cos(x - y)])

    J_exact = np.stack([f(x0, y0) for f in (df1_x, df1_y)]).T

    assert np.linalg.norm(utils.J(f1, args=[x0, y0]) - J_exact) < 1e-9
    assert utils.detJ(f1, args=[x0, y0]) - np.linalg.det(J_exact) < 1e-7

    # TODO test a rotation


def test_find_intersection():
    a = np.array([0, 1, 0])
    b = np.array([0, 0, 1])

    theta = np.random.uniform(0, np.pi / 2)
    print(1 / np.tan(theta))
    print(utils.find_intersection_params(theta, a, b))


def test_dlambda_dtheta():
    theta = np.random.uniform(0, 2 * np.pi)
    assert np.abs(utils.dlambda_dtheta(theta) - utils.D(utils.theta_to_lambda, 0, args=[theta])) < 1e-6