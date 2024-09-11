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
    
    
    
    const wgsl = device.createShaderModule({code: document.getElementById("wgsl").text});
    const pipeline = device.createRenderPipeline({
        layout: "auto",
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
    const aspect = canvas.width/canvas.height;
    var cam_const = 1.0;
    const uniformBuffer = device.createBuffer({
        size: 8, // number of bytes
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });
        const bindGroup = device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: [{
        binding: 0,
        resource: { buffer: uniformBuffer }
        }],
        });
    
        var uniforms = new Float32Array([aspect, cam_const]);
        device.queue.writeBuffer(uniformBuffer, 0, uniforms);
    
    
    
    
    
    function render(device, context, pipeline, bindGroup)
    {
        pass.setBindGroup(0, bindGroup);
        pass.setPipeline(pipeline);
        pass.draw(4);
    }
    render(device, context, pipeline, bindGroup);
    
    addEventListener("wheel", (event) => {
        cam_const *= 1.0 + 2.5e-4*event.deltaY;
        requestAnimationFrame(animate);
        });
    
    function animate()
        {
        uniforms[1] = cam_const;
        device.queue.writeBuffer(uniformBuffer, 0, uniforms);
        render(device, context, pipeline, bindGroup);
        }
    animate();
    device.queue.submit([encoder.finish()]);
    pass.end(); 
}



