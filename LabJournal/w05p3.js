"use strict"; 

window.onload = function() { main(); } 
async function main() 
{ 
    const adapter = await navigator.gpu.requestAdapter(); 
    const device = await adapter.requestDevice(); 
    const canvas = document.getElementById("webgpu-canvas");  
    const context = canvas.getContext("gpupresent") || canvas.getContext("webgpu");  
    const canvasFormat = navigator.gpu.getPreferredCanvasFormat(); 
    context.configure({device: device, format: canvasFormat,   });   
    // Create a render pass in a command buffer and submit it
    
    const aspect = canvas.width/canvas.height;
    var cam_const = 2.5;
    var uniforms = new Float32Array([aspect, cam_const]);

    let subdivs = document.querySelector('input[name="pxsubdivs"]').value**2;
    var plane_shader = document.querySelector('input[name="plane"]:checked').value;
    var triangle_shader = document.querySelector('input[name="triangle"]:checked').value;
    var use_repeat = document.querySelector('input[name="tex_lu"]:checked').value;
    var use_linear = document.querySelector('input[name="tex_filtering"]:checked').value;
    var use_texture = document.querySelector('input[name="ptex"]:checked').value;
    var shaderuniforms = new Int32Array([plane_shader, triangle_shader, use_repeat, use_linear, use_texture, subdivs]);
    const texture = await load_texture(device, "grass.jpg");
    

    let pxsize = 1/canvas.height;
    
    let jitter = new Float32Array((subdivs**2) * 2); // allowing subdivs from 1 to 10
    compute_jitters(jitter, pxsize, subdivs);

    const obj_filename = 'objectData/teapot.obj';
    const drawingInfo = await readOBJFile(obj_filename, 1, true); // file name, scale, ccw vertices

    function render()
    {
        const encoder = device.createCommandEncoder(); 
        const pass = encoder.beginRenderPass(
        { 
            colorAttachments: [{ 
            view: context.getCurrentTexture().createView(), 
            loadOp: "clear", 
            storeOp:"store",
            }]
        }); 
        // Insert render pass commands here 
        const bindGroupLayout = device.createBindGroupLayout({
            entries: [
            {
                binding: 0,
                visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT,
                buffer: {},
            },
            {
                binding: 1,
                visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT,
                buffer: {},
            },
            {
                binding: 2,
                visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT,
                buffer: {},
            },
            {
                binding: 3,
                visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT,
                buffer: {},
            },
            {
                binding: 4,
                visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT,
                buffer: {},
            },
            {
                binding: 5,
                visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT,
                buffer: {},
            },],
        });

        const wgsl = device.createShaderModule({code: document.getElementById("wgsl").text});
        const pipeline = device.createRenderPipeline({
            layout: 'auto',
            vertex: {
            module: wgsl,
            entryPoint: "main_vs",
            },
            fragment: {
            module: wgsl,
            entryPoint: "main_fs",
            targets: [{ format: canvasFormat }]
            },
            primitive: {
            topology: "triangle-strip",
            },
            });
        
        pass.setPipeline(pipeline);

        const uniformBuffer = device.createBuffer({
            size: 32, // number of bytes
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
            });

        const shaderBuffer = device.createBuffer({
            size: 32, // number of bytes
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
            });

        const jitterBuffer = device.createBuffer({
            size: jitter.byteLength,
            usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.STORAGE
            });

        const positionsBuffer = device.createBuffer({
            size: drawingInfo.vertices.byteLength,
            usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.STORAGE
            });
            

        const indicesBuffer = device.createBuffer({
            size: drawingInfo.indices.byteLength,
            usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.STORAGE
            });
            
        const normalsBuffer = device.createBuffer({
            size: drawingInfo.normals.byteLength,
            usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.STORAGE
            });

        const bindGroup = device.createBindGroup({
            layout: pipeline.getBindGroupLayout(0),
            entries: [
                {binding: 0,
                resource: { buffer: uniformBuffer },
                }, 
                {binding: 1,
                resource: { buffer: shaderBuffer },
                },
                {binding: 2, 
                resource: texture.createView() 
                },
                {binding: 3, 
                resource: { buffer: jitterBuffer }
                },
                {binding: 4, 
                resource: { buffer: positionsBuffer }
                },
                {binding: 5, 
                resource: { buffer: indicesBuffer }
                },
                {binding: 6, 
                resource: { buffer: normalsBuffer }
                },
            ],
            });
        
        

            pass.setBindGroup(0, bindGroup);
            
            device.queue.writeBuffer(uniformBuffer, 0, uniforms);
            device.queue.writeBuffer(shaderBuffer, 0, shaderuniforms);
            device.queue.writeBuffer(jitterBuffer, 0, jitter);
            device.queue.writeBuffer(positionsBuffer, 0, drawingInfo.vertices);
            device.queue.writeBuffer(indicesBuffer, 0, drawingInfo.indices);
            device.queue.writeBuffer(normalsBuffer, 0, drawingInfo.normals);
            pass.draw(4);
            pass.end(); 
            device.queue.submit([encoder.finish()]);
    }
        
    addEventListener("wheel", (event) => {
        cam_const *= 1.0 + 2.5e-4*event.deltaY;
        //requestAnimationFrame(animate);
        });
    function animate()
        {
        uniforms[1] = cam_const;
        //device.queue.writeBuffer(uniformBuffer, 0, uniforms);
        render();
        }
    render();
    
}
async function load_texture(device, filename)
{
const response = await fetch(filename);
const blob = await response.blob();
const img = await createImageBitmap(blob, { colorSpaceConversion: 'none' });
const texture = device.createTexture({
size: [img.width, img.height, 1],
format: "rgba8unorm",
usage: GPUTextureUsage.COPY_DST | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.RENDER_ATTACHMENT
});
device.queue.copyExternalImageToTexture(
{ source: img, flipY: true },
{ texture: texture },
{ width: img.width, height: img.height },
);
return texture;
}
function compute_jitters(jitter, pixelsize, subdivs)
{
    const step = pixelsize/subdivs;
    if(subdivs < 2) 
        {
            jitter[0] = 0.0;
            jitter[1] = 0.0;
        }
    else 
        {
            for(var i = 0; i < subdivs; ++i)
            for(var j = 0; j < subdivs; ++j) 
        {
            const idx = (i*subdivs + j)*2;
            jitter[idx] = (Math.random() + j)*step - pixelsize*0.5;
            jitter[idx + 1] = (Math.random() + i)*step - pixelsize*0.5;
        }
    }
}