import matplotlib.pyplot as plt
import numpy as np
from lipstick import GifMaker
from matplotlib.patches import ArrowStyle, FancyArrowPatch
from mpl_toolkits.mplot3d.axes3d import Axes3D
from mpl_toolkits.mplot3d.proj3d import proj_transform

from chick.utils.palettes import palettes
from propose.propose.poses.human36m import Human36mPose, MPIIPose


class Arrow3D(FancyArrowPatch):
    def __init__(self, x, y, z, dx, dy, dz, *args, **kwargs):
        super().__init__((0, 0), (0, 0), *args, **kwargs)
        self._xyz = (x, y, z)
        self._dxdydz = (dx, dy, dz)

    def draw(self, renderer):
        x1, y1, z1 = self._xyz
        dx, dy, dz = self._dxdydz
        x2, y2, z2 = (x1 + dx, y1 + dy, z1 + dz)

        xs, ys, zs = proj_transform((x1, x2), (y1, y2), (z1, z2), self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        super().draw(renderer)

    def do_3d_projection(self, renderer=None):
        x1, y1, z1 = self._xyz
        dx, dy, dz = self._dxdydz
        x2, y2, z2 = (x1 + dx, y1 + dy, z1 + dz)

        xs, ys, zs = proj_transform((x1, x2), (y1, y2), (z1, z2), self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))

        return np.min(zs)


def _arrow3D(ax, x, y, z, dx, dy, dz, *args, **kwargs):
    """Add an 3d arrow to an `Axes3D` instance."""

    arrow = Arrow3D(x, y, z, dx, dy, dz, *args, **kwargs)
    ax.add_artist(arrow)


setattr(Axes3D, "arrow3D", _arrow3D)


def plot_arrows(ax, lengths=[1, 2, 1]):
    arrowstyle = ArrowStyle.Fancy(head_width=5, head_length=7)
    ax.arrow3D(
        -0.53,
        -0.5,
        -1.05,
        lengths[0] / 4,
        0,
        0,
        arrowstyle=arrowstyle,
        mutation_scale=1,
        color=palettes["chick"]["black"],
    )
    ax.arrow3D(
        -0.5,
        -0.52,
        -1.05,
        0,
        lengths[1] / 4,
        0,
        arrowstyle=arrowstyle,
        mutation_scale=1,
        color=palettes["chick"]["black"],
    )
    ax.arrow3D(
        -0.5,
        -0.5,
        -1.07,
        0,
        0,
        lengths[2] / 4,
        arrowstyle=arrowstyle,
        mutation_scale=1,
        color=palettes["chick"]["black"],
    )


def plot_2D(projected_gt_3D, input_2D, samples, name, n_frames, alpha=0.1):
    projected_gt_3D = projected_gt_3D.cpu().detach().numpy().squeeze()
    input_2D = input_2D.squeeze().cpu().detach().numpy()
    # samples = np.stack([sample.cpu().detach().numpy().squeeze() for sample in samples])

    # projected_gt_3D[..., 1] = -projected_gt_3D[..., 1]
    # samples[..., 1] = -samples[..., 1]

    if n_frames == 1:
        fig = plt.figure(figsize=(4, 4), dpi=150)
        ax = fig.add_subplot(1, 1, 1)
        # ax.set_xlim(-1, 1)
        # ax.set_ylim(-1, 1)
        plt.axis("off")

        # gizmo arrows
        arrowstyle = ArrowStyle.Fancy(head_width=5, head_length=7)
        ax.add_artist(
            FancyArrowPatch(
                (-0.5, -0.97),
                (-0.5, -0.9 + 0.2),
                fc="k",
                ec="k",
                arrowstyle=arrowstyle,
                mutation_scale=1,
            )
        )
        ax.add_artist(
            FancyArrowPatch(
                (-0.52, -0.95),
                (-0.5 + 0.2, -0.95),
                fc="k",
                ec="k",
                arrowstyle=arrowstyle,
                mutation_scale=1,
            )
        )

        aux = Human36mPose(input_2D)
        aux.plot(ax, plot_type="none", c="tab:red", alpha=0.5)
        for sample in samples:
            aux = Human36mPose(sample)
            aux.plot(
                ax, plot_type="none", c=palettes["chick"]["black"], alpha=alpha, lw=1
            )

        # aux = Human36mPose(input_2D)
        # aux.plot(ax, plot_type="none", c=palettes["chick"]["red"], lw=2)
        #
        aux = Human36mPose(projected_gt_3D)
        aux.plot(ax, plot_type="none", c=palettes["candy"]["blue"], lw=2, alpha=0.5)

        plt.savefig(f"./poses/{name}.png", dpi=150, bbox_inches="tight", pad_inches=0)
        plt.close()
    else:
        with GifMaker(f"./poses/{name}.gif", fps=30) as g:
            for i in range(n_frames):
                fig = plt.figure(figsize=(4, 4), dpi=150)
                ax = fig.add_subplot(1, 1, 1)
                ax.set_xlim(-1, 1)
                ax.set_ylim(-1, 1)

                for sample in samples:  # don't know if this makes sense, tbh
                    aux = Human36mPose(sample[..., i])
                    aux.plot(ax, plot_type="none", c=palettes["chick"]["black"])

                aux = Human36mPose(projected_gt_3D[..., i])
                aux.plot(ax, plot_type="none", c=palettes["chick"]["red"])
                aux = Human36mPose(input_2D[..., i])
                aux.plot(ax, plot_type="none", c=palettes["chick"]["black"])

                g.add(fig)


def plot_3D(gt_3D, samples, name, alpha=0.1, rotation=0):
    gt_3D = gt_3D.cpu().detach().numpy()  # .squeeze()
    # samples = np.stack([sample.cpu().detach().numpy().squeeze() for sample in samples])
    samples = np.stack([sample.cpu().detach().numpy() for sample in samples])

    n_frames = gt_3D.shape[-1]
    if n_frames == 1:
        gt_3D = gt_3D[..., 0]
        samples = samples[..., 0]

        for rot in [0, 90]:
            fig = plt.figure(figsize=(4, 4), dpi=150)
            ax = fig.add_subplot(1, 1, 1, projection="3d")
            ax.set_xlim(-0.5, 0.5)
            ax.set_ylim(-0.5, 0.5)
            ax.set_zlim(-1, 1)
            ax.view_init(15, 15 - rot)

            plot_arrows(ax, lengths=[1, 2, 1] if rot == 90 else [1.4, 1, 1])

            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_zticks([])

            ax.xaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
            ax.yaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
            ax.zaxis.line.set_color((1.0, 1.0, 1.0, 0.0))

            ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
            ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))

            for sample in samples:
                aux = Human36mPose(sample)
                aux.plot(
                    ax,
                    plot_type="none",
                    c=palettes["chick"]["black"],
                    alpha=alpha,
                    lw=1,
                    zorder=10,
                )

            aux = Human36mPose(gt_3D)
            aux.plot(ax, plot_type="none", c=palettes["chick"]["red"], lw=2, zorder=20)

            plt.gca().set_box_aspect(aspect=(0.5, 0.5, 1))

            plt.savefig(
                f"./poses/{name}_rot{rot}.png",
                dpi=150,
                bbox_inches="tight",
                pad_inches=0,
            )
            plt.close()
    else:
        with GifMaker(f"./poses/{name}.gif", fps=30) as g:
            for i in range(n_frames):
                fig = plt.figure(figsize=(4, 4), dpi=150)
                ax = fig.add_subplot(1, 1, 1, projection="3d")
                ax.set_xlim(-1, 1)
                ax.set_ylim(-1, 1)
                ax.set_zlim(-1, 1)
                ax.view_init(elev=0.0, azim=0.0)
                # rot = 90
                rot = rotation
                ax.set_xlim(-0.5, 0.5)
                ax.set_ylim(-0.5, 0.5)
                ax.set_zlim(-1, 1)
                ax.view_init(15, 15 - rot)

                plot_arrows(ax, lengths=[1, 2, 1] if rot == 90 else [1.4, 1, 1])

                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_zticks([])

                ax.xaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
                ax.yaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
                ax.zaxis.line.set_color((1.0, 1.0, 1.0, 0.0))

                ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
                ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))

                plt.gca().set_box_aspect(aspect=(0.5, 0.5, 1))

                for sample in samples:  # don't know if this makes sense, tbh
                    aux = Human36mPose(sample[..., i])
                    aux.plot(ax, plot_type="none", c=palettes["chick"]["black"], lw=2)

                aux = Human36mPose(gt_3D[..., i])
                aux.plot(ax, plot_type="none", c=palettes["chick"]["red"], lw=2)

                g.add(fig)
