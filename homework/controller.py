import pystk


def control(aim_point, current_vel):
    """
    Set the Action for the low-level controller
    :param aim_point: Aim point, in screen coordinate frame [-1..1]
    :param current_vel: Current velocity of the kart
    :return: a pystk.Action (set acceleration, brake, steer, drift)
    """
    action = pystk.Action()
    """
    Your code here
    Hint: Use action.acceleration (0..1) to change the velocity. Try targeting a target_velocity (e.g. 20).
    Hint: Use action.brake to True/False to brake (optionally)
    Hint: Use action.steer to turn the kart towards the aim_point, clip the steer angle to -1..1
    Hint: You may want to use action.drift=True for wide turns (it will turn faster)
    """
    
    # Direction of the red circle
    direction = aim_point[0]
    scalingFactor = 4

    # How much do we steer towards the red circle
    if (direction > 0):
        steering = min(1, direction * scalingFactor)
    elif (direction < 0):
        steering = max(-1, direction * scalingFactor)
    else:
        steering = 0

    # Try to maintain a constant velocity
    if current_vel < 21:
        action.acceleration = 1
    else:
        action.acceleration = 0

    # Skid, if the turn is sharp
    if (direction < -0.2 or 0.2 < direction):
        if direction < 0:
            steering = -1
        elif direction > 0:
            steering = 1
        action.drift = True
    else:
        action.drift = False

    # If the red circle is too far left or right, we slow down and readjust the direction of travel
    if direction < -0.9 or 0.9 < direction:
        action.drift = False
        action.brake = True

    # Activate Nitro
    action.nitro = True
    
    # Save the steering angle
    action.steer = steering

    return action


if __name__ == '__main__':
    from .utils import PyTux
    from argparse import ArgumentParser

    def test_controller(args):
        import numpy as np
        pytux = PyTux()
        for t in args.track:
            steps, how_far = pytux.rollout(t,
                                           control,
                                           max_frames=1000,
                                           verbose=args.verbose)
            print(steps, how_far)
        pytux.close()

    parser = ArgumentParser()
    parser.add_argument('track', nargs='+')
    parser.add_argument('-v', '--verbose', action='store_true')
    args = parser.parse_args()
    test_controller(args)
