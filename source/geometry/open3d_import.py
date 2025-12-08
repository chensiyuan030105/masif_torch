# geometry/open3d_import.py
# This header file imports open3d with compatibility across versions.

import open3d as o3d

# -------- geometry / io (stable in newer Open3D) --------
if hasattr(o3d, "geometry") and hasattr(o3d, "io") and hasattr(o3d, "utility"):
    PointCloud = o3d.geometry.PointCloud
    Vector3dVector = o3d.utility.Vector3dVector
    KDTreeFlann = o3d.geometry.KDTreeFlann
    read_point_cloud = o3d.io.read_point_cloud
else:
    # Very old Open3D fallback
    PointCloud = o3d.PointCloud
    Vector3dVector = o3d.Vector3dVector
    KDTreeFlann = o3d.KDTreeFlann
    read_point_cloud = o3d.read_point_cloud

# -------- registration (API moved over time) --------
# Newer Open3D: o3d.pipelines.registration
if hasattr(o3d, "pipelines") and hasattr(o3d.pipelines, "registration"):
    reg = o3d.pipelines.registration
# Older Open3D: o3d.registration
elif hasattr(o3d, "registration"):
    reg = o3d.registration
else:
    raise AttributeError(
        "Open3D registration API not found. "
        "Your Open3D build does not expose registration modules."
    )

Feature = reg.Feature
registration_ransac_based_on_feature_matching = reg.registration_ransac_based_on_feature_matching
registration_icp = reg.registration_icp
TransformationEstimationPointToPoint = reg.TransformationEstimationPointToPoint
CorrespondenceCheckerBasedOnEdgeLength = reg.CorrespondenceCheckerBasedOnEdgeLength
CorrespondenceCheckerBasedOnDistance = reg.CorrespondenceCheckerBasedOnDistance
CorrespondenceCheckerBasedOnNormal = reg.CorrespondenceCheckerBasedOnNormal
TransformationEstimationPointToPlane = reg.TransformationEstimationPointToPlane
RANSACConvergenceCriteria = reg.RANSACConvergenceCriteria
