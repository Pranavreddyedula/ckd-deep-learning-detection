function setValues(values) {
  for (const key in values) {
    const el = document.querySelector(`[name="${key}"]`);
    if (el) el.value = values[key];
  }
}

function fillNormal() {
  setValues({
    age:30,bp:120,sg:1.025,al:0,su:0,rbc:1,pc:1,pcc:0,ba:0,
    bgr:95,bu:20,sc:0.9,sod:140,pot:4.2,hemo:15.2,pcv:45,
    wc:8000,rc:5.1,htn:0,dm:0,cad:0,appet:1,pe:0,ane:0
  });
}

function fillCKD() {
  setValues({
    age:52,bp:150,sg:1.015,al:2,su:1,rbc:0,pc:0,pcc:1,ba:1,
    bgr:180,bu:55,sc:2.3,sod:132,pot:5.4,hemo:10.8,pcv:33,
    wc:11500,rc:3.6,htn:1,dm:1,cad:0,appet:0,pe:1,ane:1
  });
}

function fillSevere() {
  setValues({
    age:68,bp:180,sg:1.005,al:4,su:3,rbc:0,pc:0,pcc:1,ba:1,
    bgr:320,bu:120,sc:6.8,sod:124,pot:6.7,hemo:7.4,pcv:22,
    wc:16000,rc:2.1,htn:1,dm:1,cad:1,appet:0,pe:1,ane:1
  });
}
