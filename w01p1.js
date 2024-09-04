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
}