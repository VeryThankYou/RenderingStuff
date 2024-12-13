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

    let subdivs = document.querySelector('input[name="pxsubdivs"]').value**2;
    var plane_shader = document.querySelector('input[name="plane"]:checked').value;
    var triangle_shader = document.querySelector('input[name="triangle"]:checked').value;
    var shaderuniforms = new Int32Array([plane_shader, triangle_shader, subdivs]);

    

    let pxsize = 1/canvas.height;
    
    let jitter = new Float32Array((subdivs**2) * 2); // allowing subdivs from 1 to 10
    compute_jitters(jitter, pxsize, subdivs);

    const obj_filename = 'objectData/CornellBoxWithBlocks.obj';
    const drawingInfo = await readOBJFile(obj_filename, 1, true); // file name, scale, ccw vertices
    let mats = new Float32Array(drawingInfo.materials.length * 2 * 4);
    for(var i = 0; i < drawingInfo.materials.length; i++)
    {
        mats[i*8] = drawingInfo.materials[i].color.r;
        mats[i*8+1] = drawingInfo.materials[i].color.g;
        mats[i*8+2] = drawingInfo.materials[i].color.b;
        mats[i*8+3] = drawingInfo.materials[i].color.a;
        mats[i*8+4] = drawingInfo.materials[i].emission.r;
        mats[i*8+5] = drawingInfo.materials[i].emission.g;
        mats[i*8+6] = drawingInfo.materials[i].emission.b;
        mats[i*8+7] = drawingInfo.materials[i].emission.a;

    }

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
        
        var objectBuffers = new Object();

        objectBuffers = build_bsp_tree(drawingInfo, device, objectBuffers);

        const uniformBuffer = device.createBuffer({
            size: 32, // number of bytes
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
            });

        const shaderBuffer = device.createBuffer({
            size: 32, // number of bytes
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
            });

        const materialsBuffer = device.createBuffer({
            size: mats.byteLength,
            usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.STORAGE
            });

        const lightIndicesBuffer = device.createBuffer({
            size: drawingInfo.light_indices.byteLength,
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
                resource: { buffer: objectBuffers.attribs }
                },
                {binding: 3, 
                resource: { buffer: objectBuffers.indices }
                },
                {binding: 4, 
                resource: { buffer: objectBuffers.colors }
                },
                {binding: 5, 
                resource: { buffer: objectBuffers.treeIds }
                },
                {binding: 6, 
                resource: { buffer: objectBuffers.bspTree }
                },
                {binding: 7, 
                resource: { buffer: objectBuffers.bspPlanes }
                },
                {binding: 8, 
                resource: { buffer: objectBuffers.aabb }
                },
                {binding: 9, 
                resource: { buffer: materialsBuffer }
                },
                {binding: 10, 
                resource: { buffer: lightIndicesBuffer }
                },
            ],
            });
        
        

            pass.setBindGroup(0, bindGroup);
            
            device.queue.writeBuffer(uniformBuffer, 0, uniforms);
            device.queue.writeBuffer(shaderBuffer, 0, shaderuniforms);
            device.queue.writeBuffer(materialsBuffer, 0, mats);
            device.queue.writeBuffer(lightIndicesBuffer, 0, drawingInfo.light_indices);


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