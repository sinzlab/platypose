from typing import Optional

import numpy as np
import numpy.typing as npt
import torch

Point2D = npt.NDArray[float]
Point3D = npt.NDArray[float]


class Camera(object):
    """
    Camera class for managing camera related operations and storing camera data.
    """

    def __init__(
        self,
        intrinsic_matrix: npt.NDArray[float],
        rotation_matrix: npt.NDArray[float],
        translation_vector: npt.NDArray[float],
        tangential_distortion: npt.NDArray[float],
        radial_distortion: npt.NDArray[float],
        frames: Optional[npt.NDArray[float]] = None,
    ):
        """
        :param intrinsic_matrix: 3x3 matrix, transforms the 3D camera coordinates to 2D homogeneous image coordinates.
        :param rotation_matrix: 3x3 matrix, describes the camera's rotation in space.
        :param translation_vector: 1x3 vector, describes the cameras location in space.
        :param tangential_distortion: 1x2 vector, describes the distortion between the lens and the image plane.
        :param radial_distortion: 1x2 or 1x3 vector, describes how light bends near the edges of the lens.
        :param frames: (optional) the mapping of corresponding frames in the video.
        """
        # check args shapes
        assert intrinsic_matrix.shape[-2:] == (
            3,
            3,
        ), "intrinsic_matrix must be a 3x3 matrix"
        assert rotation_matrix.shape[-2:] == (
            3,
            3,
        ), "rotation_matrix must be a 3x3 matrix"
        assert translation_vector.shape[-2:] == (
            1,
            3,
        ), "translation_vector must be a 1x3 vector"
        assert tangential_distortion.shape[-2:] == (
            1,
            2,
        ), "tangential_distortion must be a 1x2 vector"
        assert radial_distortion.shape[-2:] in (
            (1, 2),
            (1, 3),
        ), "radial_distortion must be a 1x2 or 1x3 vector"

        # if a batch dimension is present, remove it
        if intrinsic_matrix.dim() == 3:
            intrinsic_matrix = intrinsic_matrix.squeeze(0)
        if rotation_matrix.dim() == 3:
            rotation_matrix = rotation_matrix.squeeze(0)
        if translation_vector.dim() == 3:
            translation_vector = translation_vector.squeeze(0)
        if tangential_distortion.dim() == 3:
            tangential_distortion = tangential_distortion.squeeze(0)
        if radial_distortion.dim() == 3:
            radial_distortion = radial_distortion.squeeze(0)

        self.intrinsic_matrix = intrinsic_matrix
        self.rotation_matrix = rotation_matrix
        self.translation_vector = translation_vector
        self.tangential_distortion = tangential_distortion
        self.radial_distortion = radial_distortion
        self.frames = frames if frames is not None else torch.tensor([])
        self.device = torch.device("cpu")
        if torch.cuda.is_available():
            self.device = torch.device("cuda")

    def to_dict(self):
        return dict(
            intrinsic_matrix=self.intrinsic_matrix,
            rotation_matrix=self.rotation_matrix,
            translation_vector=self.translation_vector,
            tangential_distortion=self.tangential_distortion,
            radial_distortion=self.radial_distortion,
            frames=self.frames,
        )

    def copy(self):
        return self.__class__(**self.to_dict())

    def camera_matrix(self) -> npt.NDArray[float]:
        """
        Computes the camera matrix (M) from the rotation matrix (R) translation vector (t) and intrinsic matrix (C)
        :return: Camera matrix. M = [R | t]C
        """
        return (
            (
                torch.cat((torch.eye(3), torch.zeros(3).unsqueeze(0)), axis=0)
                @ self.intrinsic_matrix.float()
            )
            .float()
            .to(self.device)
        )

    def proj2D(
        self, points: Point3D, distort: bool = True, martinez: bool = False
    ) -> Point2D:
        """
        Computes the projection of a 3D point onto the 2D camera space
        :param points: 3D points (x, y, z) (important to have the 3d points on the last axis)
        :param distort: bool [default = True] Determines whether camera distortion should be applied to the points.
        :return: Projected 2D points (x, y)
        """
        if martinez:
            fx, fy, cx, cy, skew = self._unpack_intrinsic_matrix()

            points = points.squeeze(1)
            return torch.stack(
                [
                    torch.stack(
                        [
                            self._proj2D_martinez(
                                p,
                                self.rotation_matrix,
                                self.translation_vector.T,
                                torch.tensor([[fx], [fy]]),
                                torch.tensor([[cx], [cy]]),
                                self.radial_distortion.T,
                                self.tangential_distortion.T,
                            )
                            for p in pts
                        ]
                    )
                    for pts in points
                ]
            )

        assert points.shape[-1] == 3

        R_inv = torch.inverse(self.rotation_matrix).to(self.device)
        T_inv = -self.translation_vector.to(self.device)

        camera_matrix = self.camera_matrix()

        points = torch.matmul(points + T_inv, R_inv)

        extended_points = torch.cat(
            (points, torch.ones((*points.shape[:-1], 1), device=self.device)), axis=-1
        )

        projected_points = extended_points @ camera_matrix  # (u, v, z)
        projected_points = (
            projected_points[..., :2] / projected_points[..., 2:]
        )  # (u/z, v/z)

        if distort:
            projected_points = self.distort(projected_points)

        return projected_points

    def proj3D(self, points: Point2D, center: float) -> Point3D:
        """
        Computes the projection of a 2D point onto the 3D camera space
        :param points: 2D points (x, y) (important to have the 2d points on the last axis)
        :param center: 3D point (x, y, z) representing the root of the points.
        :return: Projected 3D points (x, y, z)
        """
        # import torch.distributions as D

        assert points.shape[-1] == 2

        R_inv = torch.inverse(self.rotation_matrix).to(self.device)
        T_inv = -self.translation_vector.to(self.device)

        center_in_cam = torch.matmul(center + T_inv, R_inv)

        extended_points = torch.cat([points, torch.ones_like(points[..., :1])], dim=-1)


        # depth = D.MultivariateNormal(loc=torch.Tensor(mean_cov['mean'])[1:], covariance_matrix=torch.Tensor(mean_cov['cov'])[1:, 1:]).sample((50, ))
        # depth = torch.cat([torch.zeros(50, 1), depth], dim=-1)

        projected_points = torch.inverse(self.intrinsic_matrix).to(self.device).float().transpose(-1, -2) @ extended_points.transpose(-1, -2).float()
        projected_points = projected_points.transpose(-1, -2)
        projected_points = projected_points / torch.norm(projected_points, dim=-1).unsqueeze(-1)
        # depth = (depth.to(self.device) + torch.norm(center_in_cam.unsqueeze(0).unsqueeze(0), dim=-1)).unsqueeze(-1)
        # projected_points = projected_points * torch.norm(center_in_cam.unsqueeze(0).unsqueeze(0), dim=-1).unsqueeze(-1)
        projected_points = projected_points * torch.norm(center_in_cam.unsqueeze(0).unsqueeze(0), dim=-1).unsqueeze(-1)
        # projected_points = projected_points * depth
        projected_points = projected_points @ self.rotation_matrix.to(self.device) + self.translation_vector.to(self.device)

        return projected_points

    def _proj2D_martinez(self, P, R, T, f, c, k, p):
        """
        Args
        P: Nx3 points in world coordinates
        R: 3x3 Camera rotation matrix
        T: 3x1 Camera translation parameters
        f: 2x1 (scalar) Camera focal length
        c: 2x1 Camera center
        k: 3x1 Camera radial distortion coefficients
        p: 2x1 Camera tangential distortion coefficients
        Returns
        Proj: Nx2 points in pixel space
        D: 1xN depth of each point in camera space
        radial: 1xN radial distortion per point
        tan: 1xN tangential distortion per point
        r2: 1xN squared radius of the projected points before distortion
        """

        # P is a matrix of 3-dimensional points
        assert len(P.shape) == 2
        assert P.shape[1] == 3

        # check shapes
        assert R.shape == (3, 3), "R must be a 3x3 matrix"
        assert T.shape == (3, 1), f"T must be a 3x1 vector, got {T.shape}"
        assert f.shape == (2, 1), f"f must be a 2x1 vector, got {f.shape}"
        assert c.shape == (2, 1), f"c must be a 2x1 vector, got {c.shape}"
        assert k.shape == (3, 1), f"k must be a 3x1 vector, got {k.shape}"
        assert p.shape == (2, 1), f"p must be a 2x1 vector, got {p.shape}"

        # cast all to cuda
        R = R.to(self.device)
        T = T.to(self.device)
        f = f.to(self.device)
        c = c.to(self.device)
        k = k.to(self.device)
        p = p.to(self.device)
        P = P.to(self.device)

        N = P.shape[0]
        X = P
        # X = R @ (P.T - T)  # rotate and translate
        XX = X[:2, :]  # / X[2, :]  # 2x16
        # r2 = XX[0, :] ** 2 + XX[1, :] ** 2  # 16,
        #
        # a = torch.tile(k, (1, N))
        # b = torch.stack([r2, r2**2, r2**3]).to(self.device)  # 3x16
        # radial = 1 + torch.einsum("ij,ij->j", a, b)  # 16,
        # tan = p[0] * XX[1, :] + p[1] * XX[0, :]  # 16,
        #
        # tm = torch.outer(torch.stack([p[1], p[0]]).reshape(-1), r2)  # 2x16
        #
        # XXX = XX * torch.tile(radial + tan, (2, 1)) + tm  # 2x16
        XXX = XX

        # Proj = (f * XXX) + c  # 2x16
        Proj = XXX
        Proj = Proj.T

        return Proj

    def distort(self, points: Point2D) -> Point2D:
        """
        Applies the radial and tangetial distortion to the pixel points.
        :param points: Undistorted 2D points in pixel space.
        :return: Distorted 2D points in pixel space.
        """
        image_points = self._pixel_to_image_points(points)

        kappa = self._radial_distortion(image_points)

        rho = self._tangential_distortion(image_points)

        distorted_image_points = image_points * kappa[..., None] + rho

        pixel_points = self._image_to_pixel_points(distorted_image_points)

        return pixel_points

    def _unpack_intrinsic_matrix(self) -> tuple[float, float, float, float, float]:
        """
        Unpacks the intrinsic matrix which is of the format
            [ fx   , 0 , 0 ]
        K = [ skew , fy, 0 ]
            [ cx   , cy, 1 ]

        :return: fx, fy, cx, cy, skew
        """
        cx = self.intrinsic_matrix[2, 0]
        cy = self.intrinsic_matrix[2, 1]
        fx = self.intrinsic_matrix[0, 0]
        fy = self.intrinsic_matrix[1, 1]
        skew = self.intrinsic_matrix[1, 0]

        return fx, fy, cx, cy, skew

    def _pixel_to_image_points(self, pixel_points: Point2D) -> Point2D:
        """
        Transforms points from pixel space to image space by computing
        x' = (x - cx) / fx
        y' = (y - cy) / fy

        :param pixel_points: 2D points in pixel space
        :return: 2D points in normalised image space
        """
        fx, fy, cx, cy, skew = self._unpack_intrinsic_matrix()

        centered_points = pixel_points - torch.tensor([[cx, cy]], device=self.device)

        y_norm = centered_points[..., 1] / fy
        x_norm = (centered_points[..., 0] - skew * y_norm) / fx

        return torch.stack([x_norm, y_norm], axis=-1)

    def _image_to_pixel_points(self, image_points: Point2D) -> Point2D:
        """
        Transforms points from image space to pixel space by computing
        x' = x * fx + cx
        y' = y * fy + cy

        :param image_points: 2D points in the normalised image space
        :return: 2D points in pixel space
        """
        fx, fy, cx, cy, skew = self._unpack_intrinsic_matrix()

        pixel_points = torch.stack(
            [
                (image_points[..., 0] * fx) + cx + (skew * image_points[..., 1]),
                image_points[..., 1] * fy + cy,
            ],
            axis=-1,
        )

        return pixel_points

    def _radial_distortion(self, image_points: Point2D) -> npt.NDArray[float]:
        """
        Occurs when light rays bend near the edge of the lens.
        Radial Distortion:
         x_dist = x(1 + k1*r^2 + k2*r^4 + k3*r^6)
         y_dist = y(1 + k1*r^2 + k2*r^4 + k3*r^6)
        where x, y are normalised in image coordinates nad translated to the optical center (x - cx) / fx, (y - cy) / fy.
        ki are the distortion coefficients.
        r^2 = x^2 + y^2
        :param image_points: 2D points in the normalised image space
        :return: (1 + k1*r^2 + k2*r^4 + k3*r^6)
        """
        r2 = image_points.__pow__(2).sum(axis=-1)
        r4 = r2**2
        r6 = r2**3

        k = torch.zeros(3)
        k[: self.radial_distortion.shape[1]] = self.radial_distortion.squeeze()

        kappa = 1 + (k[0] * r2) + (k[1] * r4) + (k[2] * r6)

        return kappa

    def _tangential_distortion(self, image_points: Point2D) -> npt.NDArray[float]:
        """
        Occurs when the lens and image plane are not in parallel.
        Tangential Distortion:
         x_dist = x + [2 * p1 * x * y + p2 * (r^2 + 2 * x^2)]
         y_dist = y + [2 * p2 * x * y + p1 * (r^2 + 2 * y^2)]

        p1 and p2 are tangential distortion coefficients.
        :param image_points: 2D points in the normalised image space
        :return: [dx, dy]
        """
        p = self.tangential_distortion.squeeze()

        r2 = image_points.__pow__(2).sum(axis=-1)

        rho = torch.stack(
            [
                2 * p[0] * image_points[..., 0] * image_points[..., 1]
                + p[1] * (r2 + 2 * image_points[..., 0] ** 2),
                2 * p[1] * image_points[..., 0] * image_points[..., 1]
                + p[0] * (r2 + 2 * image_points[..., 1] ** 2),
            ],
            axis=-1,
        )

        return rho

    def world_to_camera_view(self, points: Point3D, translate: bool = False) -> Point3D:
        """
        Transform the coordinates from world space to camera space.
        :param points: 3D points in world space
        :return: 3D points in camera space
        """
        transformed_points = self.rotation_matrix @ points[..., None]

        if translate:
            transformed_points += self.translation_vector

        return transformed_points.squeeze()

    @staticmethod
    def construct_rotation_matrix(
        alpha: float, beta: float, gamma: float
    ) -> npt.NDArray[float]:
        """
        Converts a rotation vector to a rotation matrix.
        :param rotation_vector: 3D rotation vector with euler angles: [alpha, beta, gamma] or [yaw, pitch, roll]
        :return: 3D rotation matrix

        Defines the rotation matrix as:
        R = Rz(gamma) * Ry(beta) * Rx(alpha)

        Rz = [cos(gamma), -sin(gamma), 0],
            [sin(gamma), cos(gamma), 0],
            [0, 0, 1]

        Ry = [cos(beta), 0, sin(beta)],
            [0, 1, 0],
            [-sin(beta), 0, cos(beta)]

        Rx = [1, 0, 0],
            [0, cos(alpha), -sin(alpha)],
            [0, sin(alpha), cos(alpha)]

        Reference: https://en.wikipedia.org/wiki/Rotation_matrix#General_rotations
        """
        Rz = torch.tensor(
            [
                [torch.cos(gamma), -torch.sin(gamma), 0],
                [torch.sin(gamma), torch.cos(gamma), 0],
                [0, 0, 1],
            ]
        )  # yaw

        Ry = torch.tensor(
            [
                [torch.cos(beta), 0, torch.sin(beta)],
                [0, 1, 0],
                [-torch.sin(beta), 0, torch.cos(beta)],
            ]
        )  # pitch

        Rx = torch.tensor(
            [
                [1, 0, 0],
                [0, torch.cos(alpha), -torch.sin(alpha)],
                [0, torch.sin(alpha), torch.cos(alpha)],
            ]
        )  # roll

        return Rz @ Ry @ Rx

    @staticmethod
    def construct_intrinsic_matrix(
        cx: float, cy: float, fx: float, fy: float, skew: float = 0
    ) -> npt.NDArray[float]:
        """
        Constructs the intrinsic matrix.
        :param cx: x coordinate of the optical center
        :param cy: y coordinate of the optical center
        :param fx: focal length in x direction
        :param fy: focal length in y direction
        :param skew: skew
        :return: 3x3 intrinsic matrix
        """
        return torch.tensor(
            [
                [fx, skew, cx],
                [0, fy, cy],
                [0, 0, 1],
            ]
        ).T

    def __str__(self):
        fx, fy, cx, cy, skew = self._unpack_intrinsic_matrix()
        cam_matrix = self.camera_matrix()
        return f"<{self.__class__.__name__} fx={fx:.2f} fy={fy:.2f} cx={cx:.2f} cy={cy:.2f} skew={skew:.2f}>"

    def __repr__(self):
        return self.__str__()


class DummyCamera(Camera):
    def __init__(self, *args, **kwargs):
        pass

    def proj2D(self, points: Point3D, distort: bool = True) -> Point2D:
        return points[..., [0, 1]]
