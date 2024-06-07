'use strict';

let gl;                         // The webgl context.
let surface;                    // A surface model
let shProgram;                  // A shader program
let spaceball;                  // A SimpleRotator object that lets the user rotate the view by mouse.
let surfaceLight;
let surfaceLightLine;
let surfaceAudio;
let pointLoc = [0, 0]
let camera;
let plane;
let track, webcam, texture1, texture2;
let timestamp, gyroscope, xG, yG, zG, gyroMat, alpha = 0, beta = 0, gamma = 0;
const E = 0.001
const MS2S = 1.0 / 1000.0;

function readGyroscope() {
    timestamp = Date.now();
    gyroscope = new Gyroscope();
    gyroscope.addEventListener('reading', () => {
        timestamp = Date.now();
        xG = gyroscope.x
        yG = gyroscope.y
        zG = gyroscope.z
        gyroscopeToRotationMatrix()
    })
    gyroscope.start();
}

function getRotationMatrixFromVector(rotationVector) {
    const q1 = rotationVector[0];
    const q2 = rotationVector[1];
    const q3 = rotationVector[2];
    let q0;

    if (rotationVector.length >= 4) {
        q0 = rotationVector[3];
    } else {
        q0 = 1 - q1 * q1 - q2 * q2 - q3 * q3;
        q0 = q0 > 0 ? Math.sqrt(q0) : 0;
    }
    const sq_q1 = 2 * q1 * q1;
    const sq_q2 = 2 * q2 * q2;
    const sq_q3 = 2 * q3 * q3;
    const q1_q2 = 2 * q1 * q2;
    const q3_q0 = 2 * q3 * q0;
    const q1_q3 = 2 * q1 * q3;
    const q2_q0 = 2 * q2 * q0;
    const q2_q3 = 2 * q2 * q3;
    const q1_q0 = 2 * q1 * q0;
    let R = [];
    R.push(1 - sq_q2 - sq_q3);
    R.push(q1_q2 - q3_q0);
    R.push(q1_q3 + q2_q0);
    R.push(0.0);
    R.push(q1_q2 + q3_q0);
    R.push(1 - sq_q1 - sq_q3);
    R.push(q2_q3 - q1_q0);
    R.push(0.0);
    R.push(q1_q3 - q2_q0);
    R.push(q2_q3 + q1_q0);
    R.push(1 - sq_q1 - sq_q2);
    R.push(0.0);
    R.push(0.0);
    R.push(0.0);
    R.push(0.0);
    R.push(1.0);
    return R;
}

function gyroscopeToRotationMatrix() {
    if (xG !== null) {
        let dT = (Date.now() - timestamp) * MS2S;

        let omegaMagnitude = Math.sqrt(xG * xG + yG * yG + zG * zG);

        if (omegaMagnitude > E) {
            alpha += xG * dT;
            beta += yG * dT;
            gamma += zG * dT;

            alpha = Math.min(Math.max(alpha, -Math.PI * 0.25), Math.PI * 0.25)
            document.getElementById('message').innerText = alpha
            beta = Math.min(Math.max(beta, -Math.PI * 0.25), Math.PI * 0.25)
            gamma = Math.min(Math.max(gamma, -Math.PI * 0.25), Math.PI * 0.25)
            let deltaRotationVector = [];
            deltaRotationVector.push(alpha);
            deltaRotationVector.push(beta);
            deltaRotationVector.push(gamma);

            gyroMat = getRotationMatrixFromVector(deltaRotationVector)
            timestamp = Date.now();
            // document.getElementById('message').innerText = gyroMat
        }

    }
}

function deg2rad(angle) {
    return angle * Math.PI / 180;
}

function LoadTexture() {
    let texture = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, texture);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);

    const image = new Image();
    image.crossOrigin = 'anonymus';
    image.src = "https://raw.githubusercontent.com/VolodymyrKryvonos/vggi/PA3/texture.jpg";
    image.onload = () => {
        gl.bindTexture(gl.TEXTURE_2D, texture);
        gl.texImage2D(
            gl.TEXTURE_2D,
            0,
            gl.RGBA,
            gl.RGBA,
            gl.UNSIGNED_BYTE,
            image
        );
        console.log("imageLoaded")
        draw()
    }
    return texture;
}


// Constructor
function Model(name) {
    this.name = name;
    this.iVertexBuffer = gl.createBuffer();
    this.iNormalBuffer = gl.createBuffer();
    this.iTextureBuffer = gl.createBuffer();
    this.count = 0;

    this.BufferData = function(vertices) {

        gl.bindBuffer(gl.ARRAY_BUFFER, this.iVertexBuffer);
        gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(vertices), gl.STREAM_DRAW);

        this.count = vertices.length / 3;
    }
    this.BufferDataNormal = function(normals) {

        gl.bindBuffer(gl.ARRAY_BUFFER, this.iNormalBuffer);
        gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(normals), gl.STREAM_DRAW);

    }
    this.BufferDataTexture = function(textures) {

        gl.bindBuffer(gl.ARRAY_BUFFER, this.iTextureBuffer);
        gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(textures), gl.STREAM_DRAW);

    }

    this.Draw = function() {

        gl.bindBuffer(gl.ARRAY_BUFFER, this.iVertexBuffer);
        gl.vertexAttribPointer(shProgram.iAttribVertex, 3, gl.FLOAT, false, 0, 0);
        gl.enableVertexAttribArray(shProgram.iAttribVertex);
        gl.bindBuffer(gl.ARRAY_BUFFER, this.iNormalBuffer);
        gl.vertexAttribPointer(shProgram.iAttribNormal, 3, gl.FLOAT, false, 0, 0);
        gl.enableVertexAttribArray(shProgram.iAttribNormal);
        gl.bindBuffer(gl.ARRAY_BUFFER, this.iTextureBuffer);
        gl.vertexAttribPointer(shProgram.iAttribTexture, 2, gl.FLOAT, false, 0, 0);
        gl.enableVertexAttribArray(shProgram.iAttribTexture);

        gl.drawArrays(gl.TRIANGLES, 0, this.count);
    }
    this.DrawLine = function() {

        gl.bindBuffer(gl.ARRAY_BUFFER, this.iVertexBuffer);
        gl.vertexAttribPointer(shProgram.iAttribVertex, 3, gl.FLOAT, false, 0, 0);
        gl.enableVertexAttribArray(shProgram.iAttribVertex);

        gl.drawArrays(gl.LINE_STRIP, 0, this.count);
    }
}


// Constructor
function ShaderProgram(name, program) {

    this.name = name;
    this.prog = program;

    // Location of the attribute variable in the shader program.
    this.iAttribVertex = -1;
    // Location of the uniform specifying a color for the primitive.
    this.iColor = -1;
    // Location of the uniform matrix representing the combined transformation.
    this.iModelViewProjectionMatrix = -1;

    this.Use = function() {
        gl.useProgram(this.prog);
    }
}


/* Draws a colored cube, along with a set of coordinate axes.
 * (Note that the use of the above drawPrimitive function is not an efficient
 * way to draw with WebGL.  Here, the geometry is so simple that it doesn't matter.)
 */
let audio = null,
    context,
    src,
    panner,
    highpass
function initAudioContext() {
    audio = document.getElementById('audioID');

    audio.addEventListener('play', () => {
        if (!context) {
            context = new AudioContext();
            src = context.createMediaElementSource(audio);
            panner = context.createPanner();
            highpass = context.createBiquadFilter();

            src.connect(panner);
            panner.connect(highpass);
            highpass.connect(context.destination);

            highpass.type = 'bandpass';
            highpass.Q.value = 0.1;
            highpass.frequency.value = 12345;
            context.resume();
        }
    })


    audio.addEventListener('pause', () => {
        console.log('pause');
        context.resume();
    })
    let highpassCheckbox = document.getElementById('state');
    highpassCheckbox.addEventListener('change', function() {
        if (highpassCheckbox.checked) {
            panner.disconnect();
            panner.connect(highpass);
            highpass.connect(context.destination);
        } else {
            panner.disconnect();
            panner.connect(context.destination);
        }
    });
    audio.play();
}
const { now } = Date
function draw() {

    // gyroscopeToRotationMatrix()
    gl.clearColor(0, 0, 0, 1);
    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);

    /* Set the values of the projection transformation */
    let projection = m4.perspective(Math.PI / 8, 1, 8, 12);

    /* Get the view matrix from the SimpleRotator object.*/
    let modelView = spaceball.getViewMatrix();

    let rotateToPointZero = m4.axisRotation([0.707, 0.707, 0], 0.01);
    let translateToPointZero = m4.translation(0, 0, -10);

    let matAccum0 = m4.multiply(rotateToPointZero, modelView);
    let matAccum1 = m4.multiply(translateToPointZero, matAccum0);

    /* Multiply the projection matrix times the modelview matrix to give the
       combined transformation matrix, and send that to the shader program. */
    let modelViewProjection = m4.multiply(projection, matAccum1);
    gl.bindTexture(gl.TEXTURE_2D, texture2);
    gl.texImage2D(
        gl.TEXTURE_2D,
        0,
        gl.RGBA,
        gl.RGBA,
        gl.UNSIGNED_BYTE,
        webcam
    );
    gl.uniformMatrix4fv(shProgram.iModelViewProjectionMatrix, false, m4.identity());
    plane.Draw();

    gl.clear(gl.DEPTH_BUFFER_BIT);
    gl.bindTexture(gl.TEXTURE_2D, texture1);
    // let horiz = Math.cos(Date.now() * 0.001)
    // let vertic = Math.sin(Date.now() * 0.001)
    gl.uniformMatrix4fv(shProgram.iModelViewProjectionMatrix, false, m4.translation(...getAudioVector()));
    if (panner) {
        panner.setPosition(...getAudioVector())
        highpass.Q.value =
            parseFloat(document.getElementById('q').value)
        highpass.frequency.value =
            parseFloat(document.getElementById('f').value)
    }
    surfaceAudio.Draw();
    gl.clear(gl.DEPTH_BUFFER_BIT);

    /* Draw the six faces of a cube, with different colors. */
    gl.uniform4fv(shProgram.iColor, [1, 1, 0, 1]);
    camera.ApplyLeftFrustum();
    modelViewProjection = m4.multiply(camera.mProjectionMatrix, m4.multiply(camera.mModelViewMatrix, m4.multiply(matAccum1, sensorMat)));
    gl.uniformMatrix4fv(shProgram.iModelViewProjectionMatrix, false, modelViewProjection);
    gl.colorMask(true, false, false, false);
    surface.Draw();
    gl.clear(gl.DEPTH_BUFFER_BIT);
    camera.ApplyRightFrustum();
    modelViewProjection = m4.multiply(camera.mProjectionMatrix, m4.multiply(camera.mModelViewMatrix, m4.multiply(matAccum1, sensorMat)));
    gl.uniformMatrix4fv(shProgram.iModelViewProjectionMatrix, false, modelViewProjection);
    gl.colorMask(false, true, true, false);
    surface.Draw();
    gl.colorMask(true, true, true, true);
}

function reDraw() {
    draw()
    window.requestAnimationFrame(reDraw)
}
let numStepsI = 100,
    numStepsJ = numStepsI / 2;
function CreateSurfaceData() {
    let vertexList = [];


    for (let i = 0; i < numStepsI; i++) {
        for (let j = -numStepsJ; j < numStepsJ; j++) {
            vertexList.push(...monge((i) / 10, (j) / 10))
            vertexList.push(...monge((i + 1) / 10, (j) / 10))
            vertexList.push(...monge((i) / 10, (j + 1) / 10))
            vertexList.push(...monge((i) / 10, (j + 1) / 10))
            vertexList.push(...monge((i + 1) / 10, (j) / 10))
            vertexList.push(...monge((i + 1) / 10, (j + 1) / 10))
        }
    }

    return vertexList;
}
function CreateSurfaceDataNormal() {
    let normalList = [];

    for (let i = 0; i < numStepsI; i++) {
        for (let j = -numStepsJ; j < numStepsJ; j++) {
            normalList.push(...mongeNormal((i) / 10, (j) / 10))
            normalList.push(...mongeNormal((i + 1) / 10, (j) / 10))
            normalList.push(...mongeNormal((i) / 10, (j + 1) / 10))
            normalList.push(...mongeNormal((i) / 10, (j + 1) / 10))
            normalList.push(...mongeNormal((i + 1) / 10, (j) / 10))
            normalList.push(...mongeNormal((i + 1) / 10, (j + 1) / 10))
        }
    }

    return normalList;
}
function CreateSurfaceDataTexture() {
    let normalList = [];

    for (let i = 0; i < numStepsI; i++) {
        for (let j = -numStepsJ; j < numStepsJ; j++) {
            normalList.push((i) / numStepsI, (j) / numStepsJ)
            normalList.push((i + 1) / numStepsI, (j) / numStepsJ)
            normalList.push((i) / numStepsI, (j + 1) / numStepsJ)
            normalList.push((i) / numStepsI, (j + 1) / numStepsJ)
            normalList.push((i + 1) / numStepsI, (j) / numStepsJ)
            normalList.push((i + 1) / numStepsI, (j + 1) / numStepsJ)
        }
    }

    return normalList;
}
function CreateSurfaceDataSphere() {
    let vertexList = [];

    let u = 0,
        t = 0;
    while (u < Math.PI * 2) {
        while (t < Math.PI) {
            vertexList.push(...sphere(u, t))
            vertexList.push(...sphere(u + 0.1, t))
            vertexList.push(...sphere(u, t + 0.1))
            vertexList.push(...sphere(u, t + 0.1))
            vertexList.push(...sphere(u + 0.1, t))
            vertexList.push(...sphere(u + 0.1, t + 0.1))
            t += 0.1;
        }
        t = 0;
        u += 0.1;
    }
    return vertexList;
}
const radius = 0.1;
function sphere(long, lat) {
    return [
        radius * Math.cos(long) * Math.sin(lat),
        radius * Math.sin(long) * Math.sin(lat),
        radius * Math.cos(lat)
    ]
}
const scaler = 0.1
function monge(a, t) {
    return [scaler * x(a, t), scaler * y(a, t), scaler * z(t)]
}
const e = 0.001
function mongeNormal(a, t) {
    let vert = monge(a, t)
    let vertA = monge(a + e, t)
    let vertB = monge(a, t + e)
    const n1 = [
        (vert[0] - vertA[0]) / e,
        (vert[1] - vertA[1]) / e,
        (vert[2] - vertA[2]) / e
    ]
    const n2 = [
        (vert[0] - vertB[0]) / e,
        (vert[1] - vertB[1]) / e,
        (vert[2] - vertB[2]) / e
    ]
    return m4.normalize(m4.cross(n1, n2))
}
const { PI, sin, cos } = Math;
const r = 1, theta = PI / 2, a0 = 1, aParam = 0.1;
function x(a, t) {
    let coord = r * cos(a) - (r * (a0 - a) + t * cos(theta) - aParam * t * t * sin(theta)) * sin(a)
    return coord;
}
function y(a, t) {
    let coord = r * sin(a) + (r * (a0 - a) + t * cos(theta) - aParam * t * t * sin(theta)) * cos(a)
    return coord;
}
function z(t) {
    let coord = t * sin(theta) + aParam * t * t * cos(theta);
    return coord;
}

function planeVertices() {
    const vertices = [
        [-1, -1, 0],
        [1, 1, 0],
        [1, -1, 0],
        [-1, 1, 0]
    ]
    const indices = [1, 0, 3, 0, 1, 2]
    let vertexList = []
    indices.forEach(i => {
        vertexList.push(...vertices[i])
    })
    return vertexList;
}
function planeTextures() {
    const textures = [
        [1, 1],
        [0, 0],
        [0, 1],
        [1, 0]]
    const indices = [1, 0, 3, 0, 1, 2]
    let textureList = []
    indices.forEach(i => {
        textureList.push(...textures[i])
    })
    return textureList;
}

let gui;
/* Initialize the WebGL context. Called from init() */
function initGL() {

    gui = new GUI()
    let prog = createProgram(gl, vertexShaderSource, fragmentShaderSource);

    shProgram = new ShaderProgram('Basic', prog);
    shProgram.Use();

    shProgram.iAttribVertex = gl.getAttribLocation(prog, "vertex");
    shProgram.iAttribNormal = gl.getAttribLocation(prog, "normal");
    shProgram.iAttribTexture = gl.getAttribLocation(prog, "texture");
    shProgram.iModelViewProjectionMatrix = gl.getUniformLocation(prog, "ModelViewProjectionMatrix");
    shProgram.iColor = gl.getUniformLocation(prog, "color");
    shProgram.iLightLocation = gl.getUniformLocation(prog, "lightLocation");
    shProgram.iLightDirection = gl.getUniformLocation(prog, "lightDirection");
    shProgram.iAngle = gl.getUniformLocation(prog, "angle");
    shProgram.iFocus = gl.getUniformLocation(prog, "focus");
    shProgram.iPL = gl.getUniformLocation(prog, "pointLoc");
    shProgram.iR1 = gl.getUniformLocation(prog, "r1");

    camera = new StereoCamera(4,     // Convergence
        0.25,       // Eye Separation
        1,     // Aspect Ratio
        30,       // FOV along Y in degrees
        8.0,       // Near Clipping Distance
        12.0);   // Far Clipping Distance
    gui.add(camera, 'mConvergence', 0.5, 10)
    gui.add(camera, 'mEyeSeparation', 0.01, 0.5)
    gui.add(camera, 'mFOV', 0.1, 1.5)
    gui.add(camera, 'mNearClippingDistance', 8, 10.5)

    surface = new Model('Surface');
    surface.BufferData(CreateSurfaceData());
    surface.BufferDataNormal(CreateSurfaceDataNormal());
    surface.BufferDataTexture(CreateSurfaceDataTexture());
    plane = new Model('Plane')
    plane.BufferData(planeVertices())
    plane.BufferDataNormal(planeVertices())
    plane.BufferDataTexture(planeTextures())
    surfaceAudio = new Model('Sphere')
    surfaceAudio.BufferData(CreateSurfaceDataSphere())
    surfaceAudio.BufferDataNormal(CreateSurfaceDataSphere())
    surfaceAudio.BufferDataTexture(CreateSurfaceDataSphere())

    gl.enable(gl.DEPTH_TEST);
}


/* Creates a program for use in the WebGL context gl, and returns the
 * identifier for that program.  If an error occurs while compiling or
 * linking the program, an exception of type Error is thrown.  The error
 * string contains the compilation or linking error.  If no error occurs,
 * the program identifier is the return value of the function.
 * The second and third parameters are strings that contain the
 * source code for the vertex shader and for the fragment shader.
 */
function createProgram(gl, vShader, fShader) {
    let vsh = gl.createShader(gl.VERTEX_SHADER);
    gl.shaderSource(vsh, vShader);
    gl.compileShader(vsh);
    if (!gl.getShaderParameter(vsh, gl.COMPILE_STATUS)) {
        throw new Error("Error in vertex shader:  " + gl.getShaderInfoLog(vsh));
    }
    let fsh = gl.createShader(gl.FRAGMENT_SHADER);
    gl.shaderSource(fsh, fShader);
    gl.compileShader(fsh);
    if (!gl.getShaderParameter(fsh, gl.COMPILE_STATUS)) {
        throw new Error("Error in fragment shader:  " + gl.getShaderInfoLog(fsh));
    }
    let prog = gl.createProgram();
    gl.attachShader(prog, vsh);
    gl.attachShader(prog, fsh);
    gl.linkProgram(prog);
    if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) {
        throw new Error("Link error in program:  " + gl.getProgramInfoLog(prog));
    }
    return prog;
}


/**
 * initialization function that will be called when the page has loaded
 */
let alphaS = 0, betaS = 0, gammaS = 0;
let sensorMat = m4.identity();
function init() {
    initAudioContext()
    webcam = CreateVideo()
    readGyroscope();
    window.addEventListener(
        "deviceorientation",
        (e) => {
            sensorMat = m4.multiply(
                m4.xRotation(deg2rad(e.beta)), m4.multiply(
                    m4.yRotation(deg2rad(e.gamma)),
                    m4.zRotation(deg2rad(e.alpha))))
            alphaS = e.alpha
            betaS = e.beta
            gammaS = e.gamma

        },
        true,
    );
    let canvas;
    try {
        canvas = document.getElementById("webglcanvas");
        gl = canvas.getContext("webgl");
        if (!gl) {
            throw "Browser does not support WebGL";
        }
    }
    catch (e) {
        document.getElementById("canvas-holder").innerHTML =
            "<p>Sorry, could not get a WebGL graphics context.</p>";
        return;
    }
    try {
        initGL();  // initialize the WebGL graphics context
    }
    catch (e) {
        document.getElementById("canvas-holder").innerHTML =
            "<p>Sorry, could not initialize the WebGL graphics context: " + e + "</p>";
        return;
    }

    spaceball = new TrackballRotator(canvas, draw, 0);

    reDraw();
    texture1 = LoadTexture()
    texture2 = CreateWebcamTexture2()
}

function CreateWebcamTexture2() {
    const texture = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, texture);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    return texture
}
function CreateVideo() {
    const video = document.createElement('video');
    video.setAttribute('autoplay', true);
    navigator.getUserMedia({ video: true, audio: false }, function(stream) {
        video.srcObject = stream;
        track = stream.getTracks()[0]; // this line is optional, but needs global variable track if not deleted
    }, function(e) {
        console.error('Rejected!', e);
    });
    return video;
}

function getAudioVector() {
    // Convert angles to radians
    const alphaRad = deg2rad(alphaS)
    const betaRad = deg2rad(betaS)
    const gammaRad = deg2rad(gammaS)

    // Define the initial vector along the x-axis
    let vector = [0, 1, 0];

    // Rotation around the z-axis (gamma)
    const rotZ = [
        [Math.cos(gammaRad), -Math.sin(gammaRad), 0],
        [Math.sin(gammaRad), Math.cos(gammaRad), 0],
        [0, 0, 1]
    ];
    vector = multiplyMatrixVector(rotZ, vector);

    // Rotation around the y-axis (beta)
    const rotY = [
        [Math.cos(betaRad), 0, Math.sin(betaRad)],
        [0, 1, 0],
        [-Math.sin(betaRad), 0, Math.cos(betaRad)]
    ];
    vector = multiplyMatrixVector(rotY, vector);

    // Rotation around the x-axis (alpha)
    const rotX = [
        [1, 0, 0],
        [0, Math.cos(alphaRad), -Math.sin(alphaRad)],
        [0, Math.sin(alphaRad), Math.cos(alphaRad)]
    ];
    vector = multiplyMatrixVector(rotX, vector);

    return vector;
}

function multiplyMatrixVector(matrix, vector) {
    const result = [];
    for (let i = 0; i < matrix.length; i++) {
        let sum = 0;
        for (let j = 0; j < vector.length; j++) {
            sum += matrix[i][j] * vector[j];
        }
        result.push(sum);
    }
    return result;
}