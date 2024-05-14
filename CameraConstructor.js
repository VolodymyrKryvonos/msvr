
// Constructor
function StereoCamera(
    Convergence,
    EyeSeparation,
    AspectRatio,
    FOV,
    NearClippingDistance,
    FarClippingDistance
) {
    const { tan, PI } = Math;
    this.mConvergence = Convergence;
    this.mEyeSeparation = EyeSeparation;
    this.mAspectRatio = AspectRatio;
    this.mFOV = FOV * PI / 180.0;
    this.mNearClippingDistance = NearClippingDistance;
    this.mFarClippingDistance = FarClippingDistance;
    this.mProjectionMatrix = m4.identity();
    this.mModelViewMatrix = m4.identity();

    this.ApplyLeftFrustum = function () {
        let top, bottom, left, right;

        top = this.mNearClippingDistance * tan(this.mFOV / 2);
        bottom = -top;

        const a = this.mAspectRatio * tan(this.mFOV / 2) * this.mConvergence;
        const b = a - this.mEyeSeparation / 2;
        const c = a + this.mEyeSeparation / 2;

        left = -b * this.mNearClippingDistance / this.mConvergence;
        right = c * this.mNearClippingDistance / this.mConvergence;

        this.mProjectionMatrix = m4.frustum(left, right, bottom, top,
            this.mNearClippingDistance, this.mFarClippingDistance);
        this.mModelViewMatrix = m4.translation(this.mEyeSeparation / 2, 0.0, 0.0);
    }

    this.ApplyRightFrustum = function () {
        let top, bottom, left, right;

        top = this.mNearClippingDistance * tan(this.mFOV / 2);
        bottom = -top;

        const a = this.mAspectRatio * tan(this.mFOV / 2) * this.mConvergence;
        const b = a - this.mEyeSeparation / 2;
        const c = a + this.mEyeSeparation / 2;

        left = -c * this.mNearClippingDistance / this.mConvergence;
        right = b * this.mNearClippingDistance / this.mConvergence;

        this.mProjectionMatrix = m4.frustum(left, right, bottom, top,
            this.mNearClippingDistance, this.mFarClippingDistance);
        this.mModelViewMatrix = m4.translation(-this.mEyeSeparation / 2, 0.0, 0.0);
    }
}