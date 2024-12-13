window.onload = function() { main(); } 
async function main() 
{ 
    document.write("hej");
var myHDR = new HDRImage();
    myHDR.src = 'meadow_2_4k.hdr';
    myHDR.gamma = 4.0;
    myHDR.exposure = -3.0;
  myHDR.onload = function() {
    myHDR.toHDRBlob(function(blob){
      var a = document.createElement('a');
      a.href = URL.createObjectURL(blob);
      a.download = 'meadow.RGBE.PNG';
      a.innerHTML = 'click to save';
      document.body.appendChild(a); // or a.click()
    }  )
  }
  document.write("hej2");
}