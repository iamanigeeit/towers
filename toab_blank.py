# Plot settings
import matplotlib

matplotlib.use('QtAgg')
from matplotlib.patches import Rectangle, Circle, Arc
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from functools import partial
import bisect

plt.ion()

# Settings for animation
POLE_WIDTH = 1
POLE_SPACING = 4
RING_HEIGHT = 1
RING_MIN_WIDTH = 0.8
RING_MAX_WIDTH = 1.5
RING_COLOURS = ['darkgreen', 'orange', 'yellow', 'lime', 'blue', 'cyan', 'fuchsia']
INTERVAL = 200


# Global variables
POLES = ['A', 'B', 'C']
POLESET = {'A', 'B', 'C'}


# Global functions
def find_other_pole(from_pole, to_pole):
    return (POLESET - {from_pole, to_pole}).pop()


def find_pole_with(ring, state):
    for pole, rings in state.items():
        if ring in rings:
            return pole
    raise ValueError(f'Ring {ring} not found!')


class TowersOfAhBoy:

    def __init__(self, biggest_ring=1, start_pole='A', state=None):
        self.biggest_ring = 0
        self.state = {pole: [] for pole in POLES}
        # State for animations
        self.fig, self.ax = plt.subplots(figsize=(10, 5))
        self.pole_centres = {}
        self.ring_patches = {}
        self.animation = None
        self.animation_params = []
        self.interval = INTERVAL
        self.reset(biggest_ring, start_pole, state)

    def __repr__(self):
        return str(self.state)

    def reset(self, biggest_ring=1, start_pole='A', state=None):
        # Remove all shapes in figure
        for patches in self.ring_patches.values():
            for patch in patches:
                patch.remove()
        if state is None:
            # Set biggest ring, set state to all rings on the start pole
            pass
        else:
            # Set biggest ring, set state
            pass
        self.pole_centres = self.draw_poles()
        self.ring_patches = self.draw_rings()
        self.interval = INTERVAL / self.biggest_ring

    def draw_poles(self):
        ax = self.ax
        ax.set_xlim(0 * POLE_SPACING, 3 * POLE_SPACING)  # Adjust the x-axis limits to center rods A, B, C
        ax.set_ylim(0, (self.biggest_ring + 2) * RING_HEIGHT)

        pole_centre = POLE_SPACING / 2
        xticks = []
        pole_centres = {}
        for pole in POLES:
            pole_patch = plt.Polygon(
                [
                    (pole_centre - POLE_WIDTH / 2, 0),  # Bottom-left corner
                    (pole_centre + POLE_WIDTH / 2, 0),  # Bottom-right corner
                    (pole_centre, (self.biggest_ring + 1) * RING_HEIGHT),  # Top (apex of the triangle)
                ],
                closed=True,
                facecolor="gray",
                edgecolor="black",
            )
            xticks.append(pole_centre)
            pole_centres[pole] = pole_centre
            pole_centre += POLE_SPACING
            ax.add_patch(pole_patch)
        ax.set_xticks(xticks)
        ax.set_xticklabels(POLES, fontsize=12)
        ax.set_yticklabels([])
        ax.set_aspect('equal')

        return pole_centres

    def draw_rings(self):
        ax = self.ax

        ring_patches = {}
        for pole in POLES:
            pole_centre = self.pole_centres[pole]
            for ring in self.state[pole]:
                ring_width = ring / self.biggest_ring * RING_MAX_WIDTH + RING_MIN_WIDTH
                bottom = (self.biggest_ring - ring) * RING_HEIGHT
                mid = bottom + RING_HEIGHT / 2
                top = bottom + RING_HEIGHT
                left = pole_centre - ring_width / 2
                right = left + ring_width
                color = RING_COLOURS[ring % len(RING_COLOURS)]
                rect_patch = Rectangle(
                    (left, bottom),
                    width=ring_width,
                    height=RING_HEIGHT,
                    facecolor=color,
                    edgecolor=None,
                )
                top_line = Line2D(
                    [left, right], [top, top], linewidth=1, color='black'
                )
                bottom_line = Line2D(
                    [left, right], [bottom, bottom], linewidth=1, color='black'
                )
                left_circle = Circle(
                    (left, bottom + RING_HEIGHT / 2,),
                    radius=RING_HEIGHT / 2,
                    facecolor=color,
                    edgecolor=None,
                )
                left_arc = Arc(
                    (left, mid),
                    width=RING_HEIGHT,
                    height=RING_HEIGHT,
                    theta1=90,  # Start angle (90 degrees, top of the circle)
                    theta2=270,
                    linewidth=1,
                    facecolor=color,
                    edgecolor='black',
                )
                right_circle = Circle(
                    (right, mid),
                    radius=RING_HEIGHT / 2,
                    facecolor=color,
                    edgecolor=None,
                )
                right_arc = Arc(
                    (right, mid),
                    width=RING_HEIGHT,
                    height=RING_HEIGHT,
                    theta1=270,
                    theta2=90,
                    linewidth=1,
                    facecolor=color,
                    edgecolor='black',
                )
                ax.add_patch(rect_patch)
                ax.add_line(top_line)
                ax.add_line(bottom_line)
                ax.add_patch(left_circle)
                ax.add_patch(left_arc)
                ax.add_patch(right_circle)
                ax.add_patch(right_arc)

                ring_patches[ring] = (
                    rect_patch, top_line, bottom_line, left_circle, left_arc, right_circle, right_arc
                )
        return ring_patches

    def show_state(self):
        plt.draw()

    # returns 0 for no move or 1 for single move
    def move_ring(self, ring, from_pole, to_pole, animate_now=True):
        # no move if same pole or ring is 0
        pass

        # checks: anything on pole? Is the ring on top? Is to_pole empty / has bigger ring?
        pass

        print(self.state)
        print(f'Move ring {ring} from {from_pole} to {to_pole}')

        # Move (change state)
        pass

        # Animate move
        self.animate_move(ring, from_pole, to_pole, animate_now)
        return 1

    def check_state(self):
        # Check for valid state: poles must be ABC and bigger rings must be below
        assert set(self.state.keys()) == {'A', 'B', 'C'}, 'Poles must be A,B,C only!'
        for pole, rings in self.state.items():
            for below, above in zip(rings[:-1], rings[1:]):
                assert below > above, f'Pole {pole}: ring {above} is on top of ring {below}!'

    def move_multiple_rings(self, biggest_ring, from_pole, to_pole):
        num_moves = 0
        # no move if same pole or ring is 0
        pass

        rings = list(range(biggest_ring, 0, -1))
        # Check that from_pole has the rings
        pass

        other_pole = find_other_pole(from_pole, to_pole)
        # Standard algorithm
        pass
        return num_moves


    def solve_from_reset(self, target_pole):
        from_pole = find_pole_with(self.biggest_ring, self.state)
        return self.move_multiple_rings(self.biggest_ring, from_pole, target_pole)

    # This can be considered the main method
    def solve_from_current(self, target_pole):
        self.animation_params = []
        num_moves = self.solve_from_state(self.state, self.biggest_ring, target_pole)
        frame_count = 0
        frame_starts = []
        for params in self.animation_params:
            frame_starts.append(frame_count)
            frame_count += params.pop('total_frames')
        # print(frame_starts, frame_count)
        animation_state = AnimationState()
        self.animation = FuncAnimation(
            self.fig,
            func=partial(
                multi_update,
                animation_params=self.animation_params,
                frame_starts=frame_starts,
                animation_state=animation_state,
            ),
            frames=frame_count,
            interval=self.interval, repeat=False
        )
        plt.draw()
        return num_moves

    def solve_from_state(self, state, biggest_ring, target_pole):
        num_moves = 0
        # print(substate, biggest_ring, target_pole)
        if biggest_ring == 0:
            return 0
        from_pole = find_pole_with(biggest_ring, state)
        other_pole = find_other_pole(from_pole, target_pole)
        # The algorithm is like this:
        # Create a new substate without biggest ring.
        # If from_pole = target_pole, the biggest ring is already correct. Solve the substate.
        # ElseIf biggest ring is 1, just move the ring and we're done.
        # Else, solve the substate, move biggest ring to target, create a new substate, solve the updated substate
        pass

        return num_moves

    def animate_move(self, ring, from_pole, to_pole, animate_now):
        from_centre = self.pole_centres[from_pole]
        to_centre = self.pole_centres[to_pole]
        vert_dist = (ring + 1) * RING_HEIGHT
        vert_frames = 2 * (ring + 1)
        horiz_dist = to_centre - from_centre
        horiz_frames = int(abs(horiz_dist) / POLE_SPACING * 2)
        total_frames = 2 * vert_frames + horiz_frames
        params = dict(
            ring_patches=self.ring_patches,
            ring=ring,
            horiz_dist=horiz_dist,
            horiz_frames=horiz_frames,
            vert_dist=vert_dist,
            vert_frames=vert_frames,
            total_frames=total_frames,
        )
        if animate_now:
            del params['total_frames']
            animation_state = AnimationState()
            self.animation = FuncAnimation(
                self.fig,
                func=partial(
                    update,
                    animation_state=animation_state,
                    **params
                ),
                frames=total_frames, interval=self.interval,
                repeat=False,
            )
            plt.draw()
        else:
            self.animation_params.append(params)


def adjust_xy(shape, x_move=0.0, y_move=0.0):
    if isinstance(shape, Circle) or isinstance(shape, Arc):
        old_x, old_y = shape.get_center()
        shape.set_center(xy=(old_x + x_move, old_y + y_move))
    elif isinstance(shape, Rectangle):
        old_x, old_y = shape.get_xy()
        shape.set_xy(xy=(old_x + x_move, old_y + y_move))
    elif isinstance(shape, Line2D):
        old_xs, old_ys = shape.get_xdata(), shape.get_ydata()
        shape.set_xdata([old_x + x_move for old_x in old_xs])
        shape.set_ydata([old_y + y_move for old_y in old_ys])


def adjust_all_xy(shapes, x_move=0.0, y_move=0.0):
    for shape in shapes:
        adjust_xy(shape, x_move, y_move)


def update(frame, ring_patches, ring, horiz_dist, horiz_frames, vert_dist, vert_frames, animation_state, curr_move=0):
    shapes = ring_patches[ring]
    if frame == 0 and curr_move == 0:
        if animation_state.new:
            adjust_all_xy(shapes, y_move=vert_dist / vert_frames)
            animation_state.new = False
    elif frame < vert_frames:
        adjust_all_xy(shapes, y_move=vert_dist / vert_frames)
    elif frame < vert_frames + horiz_frames:
        adjust_all_xy(shapes, x_move=horiz_dist / horiz_frames)
    else:
        adjust_all_xy(shapes, y_move=-vert_dist / vert_frames)
    return shapes


def multi_update(global_frame, animation_params, frame_starts, animation_state):
    curr_move = bisect.bisect_right(frame_starts, global_frame) - 1
    params = animation_params[curr_move]
    frame = global_frame - frame_starts[curr_move]
    return update(frame, curr_move=curr_move, animation_state=animation_state, **params)


class AnimationState:
    new = True

