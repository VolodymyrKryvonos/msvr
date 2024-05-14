'use strict';

let gl;                         // The webgl context.
let surface;                    // A surface model
let shProgram;                  // A shader program
let spaceball;                  // A SimpleRotator object that lets the user rotate the view by mouse.
let surfaceLight;
let surfaceLightLine;
let pointLoc = [0, 0]
let camera;

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
}


// Constructor
function Model(name) {
    this.name = name;
    this.iVertexBuffer = gl.createBuffer();
    this.iNormalBuffer = gl.createBuffer();
    this.iTextureBuffer = gl.createBuffer();
    this.count = 0;

    this.BufferData = function (vertices) {

        gl.bindBuffer(gl.ARRAY_BUFFER, this.iVertexBuffer);
        gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(vertices), gl.STREAM_DRAW);

        this.count = vertices.length / 3;
    }
    this.BufferDataNormal = function (normals) {

        gl.bindBuffer(gl.ARRAY_BUFFER, this.iNormalBuffer);
        gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(normals), gl.STREAM_DRAW);

    }
    this.BufferDataTexture = function (textures) {

        gl.bindBuffer(gl.ARRAY_BUFFER, this.iTextureBuffer);
        gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(textures), gl.STREAM_DRAW);

    }

    this.Draw = function () {

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
    this.DrawLine = function () {

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

    this.Use = function () {
        gl.useProgram(this.prog);
    }
}


/* Draws a colored cube, along with a set of coordinate axes.
 * (Note that the use of the above drawPrimitive function is not an efficient
 * way to draw with WebGL.  Here, the geometry is so simple that it doesn't matter.)
 */
const { now } = Date
function draw() {
    gl.clearColor(0, 0, 0, 1);
    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);

    /* Set the values of the projection transformation */
    let projection = m4.perspective(Math.PI / 8, 1, 8, 12);

    /* Get the view matrix from the SimpleRotator object.*/
    let modelView = spaceball.getViewMatrix();

    let rotateToPointZero = m4.axisRotation([0.707, 0.707, 0], 0.7);
    let translateToPointZero = m4.translation(0, 0, -10);

    let matAccum0 = m4.multiply(rotateToPointZero, modelView);
    let matAccum1 = m4.multiply(translateToPointZero, matAccum0);

    /* Multiply the projection matrix times the modelview matrix to give the
       combined transformation matrix, and send that to the shader program. */
    let modelViewProjection = m4.multiply(projection, matAccum1);

    gl.uniformMatrix4fv(shProgram.iModelViewProjectionMatrix, false, modelViewProjection);

    /* Draw the six faces of a cube, with different colors. */
    gl.uniform4fv(shProgram.iColor, [1, 1, 0, 1]);
    camera.ApplyLeftFrustum();
    modelViewProjection = m4.multiply(camera.mProjectionMatrix, m4.multiply(camera.mModelViewMatrix, matAccum1));
    gl.uniformMatrix4fv(shProgram.iModelViewProjectionMatrix, false, modelViewProjection);
    gl.colorMask(true, false, false, false);
    surface.Draw();
    gl.clear(gl.DEPTH_BUFFER_BIT);
    camera.ApplyRightFrustum();
    modelViewProjection = m4.multiply(camera.mProjectionMatrix, m4.multiply(camera.mModelViewMatrix, matAccum1));
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
    surfaceLight = new Model();
    surfaceLight.BufferData(CreateSurfaceDataSphere());
    surfaceLight.BufferDataNormal(CreateSurfaceDataSphere());
    surfaceLight.BufferDataTexture(CreateSurfaceDataSphere());
    surfaceLightLine = new Model();
    surfaceLightLine.BufferData([0, 0, 0, 1, 1, 1]);

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
function init() {
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
    LoadTexture()
}
