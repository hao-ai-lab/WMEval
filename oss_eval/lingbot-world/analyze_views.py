import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Load the files
intrinsics = np.load('examples/billiards_game_0/intrinsics.npy')
poses = np.load('examples/billiards_game_0/poses.npy')

print("=" * 70)
print("CAMERA INTRINSICS (Internal Parameters)")
print("=" * 70)
print(f"Shape: {intrinsics.shape} → (161 frames, 4 parameters)")
print(f"\nFirst frame: {intrinsics[0]}")
print(f"Last frame:  {intrinsics[-1]}")
print(f"All identical? {np.all(intrinsics == intrinsics[0])}")

print("\n📐 Parameter meanings:")
print(f"  fx = {intrinsics[0,0]:6.1f}  → Focal length X (horizontal zoom)")
print(f"  fy = {intrinsics[0,1]:6.1f}  → Focal length Y (vertical zoom)")
print(f"  cx = {intrinsics[0,2]:6.1f}  → Principal point X (center X)")
print(f"  cy = {intrinsics[0,3]:6.1f}  → Principal point Y (center Y)")

print("\n📺 For 480×832 video:")
print(f"  Width center: 832/2 = 416 ✓")
print(f"  Height center: 480/2 = 240 ✓")

print("\n" + "=" * 70)
print("CAMERA POSES (External Parameters - Position & Orientation)")
print("=" * 70)
print(f"Shape: {poses.shape} → (161 frames, 4×4 transformation matrix)")
print(f"\nFirst frame pose:\n{poses[0]}")
print(f"\nAll identical? {np.all(poses == poses[0])}")

# Check if identity
is_identity = np.allclose(poses[0], np.eye(4))
print(f"Is identity matrix? {is_identity}")

if is_identity:
    print("\n🎥 Identity pose means:")
    print("  ✓ Camera at world origin (0, 0, 0)")
    print("  ✓ No rotation (aligned with world axes)")
    print("  ✓ Looking down -Z axis (standard convention)")
    print("  ✓ X points right, Y points down, Z points forward")

# Extract camera positions across all frames
camera_positions = poses[:, :3, 3]
print(f"\n📍 Camera position range:")
print(f"  X: [{camera_positions[:,0].min():.3f}, {camera_positions[:,0].max():.3f}]")
print(f"  Y: [{camera_positions[:,1].min():.3f}, {camera_positions[:,1].max():.3f}]")
print(f"  Z: [{camera_positions[:,2].min():.3f}, {camera_positions[:,2].max():.3f}]")

# Create visualization
fig = plt.figure(figsize=(16, 5))

# Plot 1: Intrinsics parameters over frames
ax1 = fig.add_subplot(141)
frames = np.arange(len(intrinsics))
ax1.plot(frames, intrinsics[:, 0], label='fx (focal X)', linewidth=2)
ax1.plot(frames, intrinsics[:, 1], label='fy (focal Y)', linewidth=2)
ax1.plot(frames, intrinsics[:, 2], label='cx (center X)', linewidth=2)
ax1.plot(frames, intrinsics[:, 3], label='cy (center Y)', linewidth=2)
ax1.set_xlabel('Frame Number')
ax1.set_ylabel('Parameter Value (pixels)')
ax1.set_title('Camera Intrinsics\n(All Static)')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Camera trajectory (will be a single point)
ax2 = fig.add_subplot(142, projection='3d')
ax2.scatter(camera_positions[:, 0], 
            camera_positions[:, 1], 
            camera_positions[:, 2], 
            c=np.arange(len(camera_positions)), 
            cmap='viridis', s=50, marker='o')
ax2.scatter([0], [0], [0], c='red', s=200, marker='*', 
            label='Camera (static at origin)')
ax2.set_xlabel('X Position')
ax2.set_ylabel('Y Position')
ax2.set_zlabel('Z Position')
ax2.set_title('Camera Position\n(Static at Origin)')
ax2.legend()
ax2.grid(True)

# Plot 3: Camera orientation axes
ax3 = fig.add_subplot(143, projection='3d')
origin = np.array([0, 0, 0])
R = poses[0, :3, :3]  # Rotation matrix

# Camera axes
scale = 1.5
x_axis = R @ np.array([scale, 0, 0])
y_axis = R @ np.array([0, scale, 0])
z_axis = R @ np.array([0, 0, scale])

ax3.quiver(*origin, *x_axis, color='red', arrow_length_ratio=0.2, 
           linewidth=3, label='X (Right)', alpha=0.8)
ax3.quiver(*origin, *y_axis, color='green', arrow_length_ratio=0.2, 
           linewidth=3, label='Y (Down)', alpha=0.8)
ax3.quiver(*origin, *z_axis, color='blue', arrow_length_ratio=0.2, 
           linewidth=3, label='Z (Forward)', alpha=0.8)
ax3.scatter([0], [0], [0], c='black', s=150, marker='o', 
            label='Camera', zorder=10)

ax3.set_xlim([-2, 2])
ax3.set_ylim([-2, 2])
ax3.set_zlim([-2, 2])
ax3.set_xlabel('X')
ax3.set_ylabel('Y')
ax3.set_zlabel('Z')
ax3.set_title('Camera Coordinate System\n(Identity Orientation)')
ax3.legend()

# Plot 4: Image plane visualization
ax4 = fig.add_subplot(144)
img_height, img_width = 480, 832
cx, cy = intrinsics[0, 2], intrinsics[0, 3]
ax4.add_patch(plt.Rectangle((0, 0), img_width, img_height, 
              fill=False, edgecolor='blue', linewidth=2))
ax4.plot(cx, cy, 'r*', 
         markersize=20, label=f'Principal Point ({cx:.0f}, {cy:.0f})')
ax4.set_xlim(-50, img_width + 50)
ax4.set_ylim(-50, img_height + 50)
ax4.set_xlabel('Width (pixels)')
ax4.set_ylabel('Height (pixels)')
ax4.set_title('Image Plane\n480×832 pixels')
ax4.legend()
ax4.grid(True, alpha=0.3)
ax4.set_aspect('equal')
ax4.invert_yaxis()  # Image coordinates: Y increases downward

plt.tight_layout()
plt.savefig('examples/billiards_game_0/camera_analysis.png', 
            dpi=150, bbox_inches='tight')
print(f"\n✅ Visualization saved to: examples/billiards_game_0/camera_analysis.png")
plt.show()