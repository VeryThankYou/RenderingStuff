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
    var cam_const = 1.0;
    var uniforms = new Float32Array([aspect, cam_const]);


    var plane_shader = document.querySelector('input[name="plane"]:checked').value;
    var triangle_shader = document.querySelector('input[name="triangle"]:checked').value;
    var sphere_shader = document.querySelector('input[name="sphere"]:checked').value;
    var shaderuniforms = new Int32Array([plane_shader, triangle_shader, sphere_shader]);
    
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

        const bindGroup = device.createBindGroup({
            layout: pipeline.getBindGroupLayout(0),
            entries: [
                {binding: 0,
                resource: { buffer: uniformBuffer },
                }, 
                {binding: 1,
                resource: { buffer: shaderBuffer },
                }],
            });
        
        

            pass.setBindGroup(0, bindGroup);
            
            device.queue.writeBuffer(uniformBuffer, 0, uniforms);
            device.queue.writeBuffer(shaderBuffer, 0, shaderuniforms);
            pass.draw(4);
            pass.end(); 
            device.queue.submit([encoder.finish()]);
    }
        
    addEventListener("wheel", (event) => {
        cam_const *= 1.0 + 2.5e-4*event.deltaY;
        requestAnimationFrame(animate);
        });
    function animate()
        {
        uniforms[1] = cam_const;
        //device.queue.writeBuffer(uniformBuffer, 0, uniforms);
        render();
        }
    render();
    
}