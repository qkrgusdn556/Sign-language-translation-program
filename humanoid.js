import * as THREE from "https://cdn.jsdelivr.net/npm/three@0.180.0/build/three.module.js";

export function createHuman(scene) {
  const body = new THREE.Group();
  scene.add(body);
  const material = (color) => new THREE.MeshStandardMaterial({ color, roughness: 0.76 });
  const skin = material(0xdca88b);
  const hair = material(0x292322);
  const shirt = material(0x597c94);
  const trim = material(0x365769);
  const white = material(0xf6f1e8);
  const iris = material(0x44382d);
  const lip = material(0xa96861);
  const sphere = new THREE.SphereGeometry(1, 32, 24);
  function oval(parent, mat, position, scale) {
    const mesh = new THREE.Mesh(sphere, mat);
    mesh.position.set(...position);
    mesh.scale.set(...scale);
    parent.add(mesh);
    return mesh;
  }
  oval(body, shirt, [0, -0.66, -0.12], [0.68, 0.91, 0.33]);
  oval(body, shirt, [0, -0.12, -0.13], [0.88, 0.32, 0.34]);
  oval(body, trim, [0, -1.4, -0.12], [0.55, 0.11, 0.28]);
  oval(body, skin, [0, 0.24, -0.05], [0.16, 0.28, 0.17]);
  const collar = new THREE.Mesh(new THREE.TorusGeometry(0.19, 0.035, 12, 40), trim);
  collar.rotation.x = Math.PI / 2;
  collar.position.set(0, 0.08, -0.06);
  body.add(collar);

  const head = new THREE.Group();
  head.position.set(0, 0.82, 0);
  body.add(head);
  oval(head, skin, [0, 0, 0], [0.35, 0.47, 0.31]);
  oval(head, skin, [0, -0.23, 0.055], [0.27, 0.25, 0.25]);
  for (const side of [-1, 1]) {
    oval(head, skin, [side * 0.345, -0.025, 0], [0.073, 0.12, 0.07]);
    oval(head, lip, [side * 0.37, -0.025, 0.052], [0.025, 0.066, 0.018]);
    oval(head, white, [side * 0.135, 0.045, 0.279], [0.087, 0.045, 0.028]);
    oval(head, iris, [side * 0.135, 0.045, 0.304], [0.032, 0.034, 0.012]);
    oval(head, hair, [side * 0.135, 0.045, 0.315], [0.015, 0.022, 0.006]);
    oval(head, white, [side * 0.135 - 0.009, 0.057, 0.321], [0.008, 0.009, 0.004]);
    const brow = oval(head, hair, [side * 0.137, 0.14, 0.281], [0.095, 0.019, 0.023]);
    brow.rotation.z = side * -0.09;
  }
  oval(head, skin, [0, -0.033, 0.3], [0.047, 0.105, 0.06]);
  oval(head, skin, [0, -0.101, 0.343], [0.063, 0.04, 0.047]);
  const smile = new THREE.CatmullRomCurve3([
    new THREE.Vector3(-0.108, -0.214, 0.277),
    new THREE.Vector3(0, -0.232, 0.298),
    new THREE.Vector3(0.108, -0.214, 0.277),
  ]);
  head.add(new THREE.Mesh(new THREE.TubeGeometry(smile, 20, 0.012, 8, false), lip));
  const cap = new THREE.Mesh(new THREE.SphereGeometry(1, 40, 24, 0, Math.PI * 2, 0, 1.42), hair);
  cap.scale.set(0.366, 0.48, 0.328);
  cap.position.set(0, 0.025, -0.025);
  head.add(cap);
  const fringe = oval(head, hair, [-0.095, 0.288, 0.237], [0.265, 0.115, 0.105]);
  fringe.rotation.z = 0.22;
  for (const side of [-1, 1]) oval(head, hair, [side * 0.317, 0.115, -0.01], [0.05, 0.18, 0.2]);

  const arms = [-1, 1].map((side) => {
    const shoulder = new THREE.Vector3(side * 0.68, -0.12, -0.08);
    const upper = segment(shirt, 0.16, 0.19);
    const lower = segment(skin, 0.09, 0.125);
    const elbow = oval(scene, skin, [0, 0, 0], [0.125, 0.125, 0.125]);
    scene.add(upper, lower);
    return { side, shoulder, upper, lower, elbow };
  });
  return { body, skin, arms };
}

export function segment(material, top, bottom = top) {
  return new THREE.Mesh(new THREE.CylinderGeometry(top, bottom, 1, 12), material);
}

export function connect(mesh, start, end) {
  const delta = new THREE.Vector3().subVectors(end, start);
  mesh.position.copy(start).add(end).multiplyScalar(0.5);
  mesh.scale.y = Math.max(delta.length(), 0.001);
  mesh.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), delta.normalize());
}

export function poseArms(arms, rigs, targets = null) {
  arms.forEach((arm, index) => {
    const wrist = rigs[index].group.position;
    const target = targets?.[index];
    const elbow = target
      ? new THREE.Vector3(target.elbow.x, target.elbow.y, target.elbow.z)
      : new THREE.Vector3(arm.side * 0.96, Math.min(-0.66, wrist.y - 0.3), 0.22);
    connect(arm.upper, arm.shoulder, elbow);
    connect(arm.lower, elbow, wrist);
    arm.elbow.position.copy(elbow);
  });
}
