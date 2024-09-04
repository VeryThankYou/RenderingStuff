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
    pass.setPipeline(pipeline);
    pass.draw(4);
    pass.end(); 
    
    device.queue.submit([encoder.finish()]);
}