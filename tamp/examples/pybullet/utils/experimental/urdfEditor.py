import pybullet as p

UNKNOWN_FILE = 'unknown_file'

class UrdfInertial(object):
    def __init__(self):
        self.mass = 1
        self.inertia_xxyyzz = [7, 8, 9]
        self.origin_rpy = [1, 2, 3]
        self.origin_xyz = [4, 5, 6]


class UrdfContact(object):
    def __init__(self):
        self.lateral_friction = 1
        self.rolling_friction = 0
        self.spinning_friction = 0


class UrdfLink(object):
    def __init__(self):
        self.link_name = "dummy"
        self.urdf_inertial = UrdfInertial()
        self.urdf_visual_shapes = []
        self.urdf_collision_shapes = []


class UrdfVisual(object):
    def __init__(self):
        self.origin_rpy = [1, 2, 3]
        self.origin_xyz = [4, 5, 6]
        self.geom_type = p.GEOM_BOX
        self.geom_radius = 1
        self.geom_extents = [7, 8, 9]
        #self.geom_length = [10]
        self.geom_length = 10
        self.geom_meshfilename = "meshfile"
        self.geom_meshscale = [1, 1, 1]
        self.material_rgba = [1, 0, 0, 1]
        self.material_name = ""


class UrdfCollision(object):
    def __init__(self):
        self.origin_rpy = [1, 2, 3]
        self.origin_xyz = [4, 5, 6]
        self.geom_type = p.GEOM_BOX
        self.geom_radius = 1
        self.geom_length = 2
        self.geom_extents = [7, 8, 9]
        self.geom_meshfilename = "meshfile"
        self.geom_meshscale = [1, 1, 1]


class UrdfJoint(object):
    def __init__(self):
        self.link = UrdfLink()
        self.joint_name = "joint_dummy"
        self.joint_type = p.JOINT_REVOLUTE
        self.joint_lower_limit = 0
        self.joint_upper_limit = -1
        self.parent_name = "parentName"
        self.child_name = "childName"
        self.joint_origin_xyz = [1, 2, 3]
        self.joint_origin_rpy = [1, 2, 3]
        self.joint_axis_xyz = [1, 2, 3]


class UrdfEditor(object):
    def __init__(self):
        self.initialize()

    def initialize(self):
        self.urdfLinks = []
        self.urdfJoints = []
        self.robotName = ""
        self.linkNameToIndex = {}
        self.jointTypeToName = {p.JOINT_FIXED: "JOINT_FIXED", \
                                p.JOINT_REVOLUTE: "JOINT_REVOLUTE", \
                                p.JOINT_PRISMATIC: "JOINT_PRISMATIC"}

    def convertLinkFromMultiBody(self, bodyUid, linkIndex, urdfLink, physicsClientId):
        dyn = p.getDynamicsInfo(bodyUid, linkIndex, physicsClientId=physicsClientId)
        urdfLink.urdf_inertial.mass = dyn[0]
        urdfLink.urdf_inertial.inertia_xxyyzz = dyn[2]
        # todo
        urdfLink.urdf_inertial.origin_xyz = dyn[3]
        urdfLink.urdf_inertial.origin_rpy = p.getEulerFromQuaternion(dyn[4])

        visualShapes = p.getVisualShapeData(bodyUid, physicsClientId=physicsClientId)
        matIndex = 0
        for v in visualShapes:
            if (v[1] == linkIndex):
                urdfVisual = UrdfVisual()
                urdfVisual.geom_type = v[2]
                if (v[2] == p.GEOM_BOX):
                    urdfVisual.geom_extents = v[3]
                if (v[2] == p.GEOM_SPHERE):
                    urdfVisual.geom_radius = v[3][0]
                if (v[2] == p.GEOM_MESH):
                    urdfVisual.geom_meshfilename = v[4].decode("utf-8")
                    if urdfVisual.geom_meshfilename == UNKNOWN_FILE:
                        continue
                    urdfVisual.geom_meshscale = v[3]
                if (v[2] == p.GEOM_CYLINDER):
                    urdfVisual.geom_length = v[3][0]
                    urdfVisual.geom_radius = v[3][1]
                if (v[2] == p.GEOM_CAPSULE):
                    urdfVisual.geom_length = v[3][0]
                    urdfVisual.geom_radius = v[3][1]
                urdfVisual.origin_xyz = v[5]
                urdfVisual.origin_rpy = p.getEulerFromQuaternion(v[6])
                urdfVisual.material_rgba = v[7]
                name = 'mat_{}_{}'.format(linkIndex, matIndex)
                urdfVisual.material_name = name
                urdfLink.urdf_visual_shapes.append(urdfVisual)
                matIndex = matIndex + 1

        collisionShapes = p.getCollisionShapeData(bodyUid, linkIndex, physicsClientId=physicsClientId)
        for v in collisionShapes:
            urdfCollision = UrdfCollision()
            urdfCollision.geom_type = v[2]
            if (v[2] == p.GEOM_BOX):
                urdfCollision.geom_extents = v[3]
            if (v[2] == p.GEOM_SPHERE):
                urdfCollision.geom_radius = v[3][0]
            if (v[2] == p.GEOM_MESH):
                urdfCollision.geom_meshfilename = v[4].decode("utf-8")
                if urdfCollision.geom_meshfilename == UNKNOWN_FILE:
                    continue
                urdfCollision.geom_meshscale = v[3]
            if (v[2] == p.GEOM_CYLINDER):
                urdfCollision.geom_length = v[3][0]
                urdfCollision.geom_radius = v[3][1]
            if (v[2] == p.GEOM_CAPSULE):
                urdfCollision.geom_length = v[3][0]
                urdfCollision.geom_radius = v[3][1]
            pos, orn = p.multiplyTransforms(dyn[3], dyn[4], v[5], v[6])
            urdfCollision.origin_xyz = pos
            urdfCollision.origin_rpy = p.getEulerFromQuaternion(orn)
            urdfLink.urdf_collision_shapes.append(urdfCollision)

    def initializeFromBulletBody(self, bodyUid, physicsClientId):
        self.initialize()

        # always create a base link
        baseLink = UrdfLink()
        baseLinkIndex = -1
        self.convertLinkFromMultiBody(bodyUid, baseLinkIndex, baseLink, physicsClientId)
        baseLink.link_name = p.getBodyInfo(bodyUid, physicsClientId=physicsClientId)[0].decode("utf-8")
        self.linkNameToIndex[baseLink.link_name] = len(self.urdfLinks)
        self.urdfLinks.append(baseLink)

        # optionally create child links and joints
        for j in range(p.getNumJoints(bodyUid, physicsClientId=physicsClientId)):
            jointInfo = p.getJointInfo(bodyUid, j, physicsClientId=physicsClientId)
            urdfLink = UrdfLink()
            self.convertLinkFromMultiBody(bodyUid, j, urdfLink, physicsClientId)
            urdfLink.link_name = jointInfo[12].decode("utf-8")
            self.linkNameToIndex[urdfLink.link_name] = len(self.urdfLinks)
            self.urdfLinks.append(urdfLink)

            urdfJoint = UrdfJoint()
            urdfJoint.link = urdfLink
            urdfJoint.joint_name = jointInfo[1].decode("utf-8")
            urdfJoint.joint_type = jointInfo[2]
            urdfJoint.joint_axis_xyz = jointInfo[13]
            orgParentIndex = jointInfo[16]
            if (orgParentIndex < 0):
                urdfJoint.parent_name = baseLink.link_name
            else:
                parentJointInfo = p.getJointInfo(bodyUid, orgParentIndex, physicsClientId=physicsClientId)
                urdfJoint.parent_name = parentJointInfo[12].decode("utf-8")
            urdfJoint.child_name = urdfLink.link_name

            # todo, compensate for inertia/link frame offset

            dynChild = p.getDynamicsInfo(bodyUid, j, physicsClientId=physicsClientId)
            childInertiaPos = dynChild[3]
            childInertiaOrn = dynChild[4]
            parentCom2JointPos = jointInfo[14]
            parentCom2JointOrn = jointInfo[15]
            tmpPos, tmpOrn = p.multiplyTransforms(childInertiaPos, childInertiaOrn, parentCom2JointPos,
                                                  parentCom2JointOrn)
            tmpPosInv, tmpOrnInv = p.invertTransform(tmpPos, tmpOrn)
            dynParent = p.getDynamicsInfo(bodyUid, orgParentIndex, physicsClientId=physicsClientId)
            parentInertiaPos = dynParent[3]
            parentInertiaOrn = dynParent[4]

            pos, orn = p.multiplyTransforms(parentInertiaPos, parentInertiaOrn, tmpPosInv, tmpOrnInv)
            pos, orn_unused = p.multiplyTransforms(parentInertiaPos, parentInertiaOrn, parentCom2JointPos, [0, 0, 0, 1])

            urdfJoint.joint_origin_xyz = pos
            urdfJoint.joint_origin_rpy = p.getEulerFromQuaternion(orn)

            self.urdfJoints.append(urdfJoint)

    def writeInertial(self, file, urdfInertial, precision=5):
        file.write("\t\t<inertial>\n")
        str = '\t\t\t<origin rpy=\"{:.{prec}f} {:.{prec}f} {:.{prec}f}\" xyz=\"{:.{prec}f} {:.{prec}f} {:.{prec}f}\"/>\n'.format( \
            urdfInertial.origin_rpy[0], urdfInertial.origin_rpy[1], urdfInertial.origin_rpy[2], \
            urdfInertial.origin_xyz[0], urdfInertial.origin_xyz[1], urdfInertial.origin_xyz[2], prec=precision)
        file.write(str)
        str = '\t\t\t<mass value=\"{:.{prec}f}\"/>\n'.format(urdfInertial.mass, prec=precision)
        file.write(str)
        str = '\t\t\t<inertia ixx=\"{:.{prec}f}\" ixy=\"0\" ixz=\"0\" iyy=\"{:.{prec}f}\" iyz=\"0\" izz=\"{:.{prec}f}\"/>\n'.format( \
            urdfInertial.inertia_xxyyzz[0], \
            urdfInertial.inertia_xxyyzz[1], \
            urdfInertial.inertia_xxyyzz[2], prec=precision)
        file.write(str)
        file.write("\t\t</inertial>\n")

    def writeVisualShape(self, file, urdfVisual, precision=5):
        file.write("\t\t<visual>\n")
        str = '\t\t\t<origin rpy="{:.{prec}f} {:.{prec}f} {:.{prec}f}" xyz="{:.{prec}f} {:.{prec}f} {:.{prec}f}"/>\n'.format( \
            urdfVisual.origin_rpy[0], urdfVisual.origin_rpy[1], urdfVisual.origin_rpy[2],
            urdfVisual.origin_xyz[0], urdfVisual.origin_xyz[1], urdfVisual.origin_xyz[2], prec=precision)
        file.write(str)
        file.write("\t\t\t<geometry>\n")
        if urdfVisual.geom_type == p.GEOM_BOX:
            str = '\t\t\t\t<box size=\"{:.{prec}f} {:.{prec}f} {:.{prec}f}\"/>\n'.format(urdfVisual.geom_extents[0], \
                                                                                         urdfVisual.geom_extents[1],
                                                                                         urdfVisual.geom_extents[2],
                                                                                         prec=precision)
            file.write(str)
        if urdfVisual.geom_type == p.GEOM_SPHERE:
            str = '\t\t\t\t<sphere radius=\"{:.{prec}f}\"/>\n'.format(urdfVisual.geom_radius, \
                                                                      prec=precision)
            file.write(str)
        if urdfVisual.geom_type == p.GEOM_MESH:
            str = '\t\t\t\t<mesh filename=\"{}\"/>\n'.format(urdfVisual.geom_meshfilename, \
                                                             prec=precision)
            file.write(str)
        if urdfVisual.geom_type == p.GEOM_CYLINDER:
            str = '\t\t\t\t<cylinder length=\"{:.{prec}f}\" radius=\"{:.{prec}f}\"/>\n'.format( \
                urdfVisual.geom_length, urdfVisual.geom_radius, prec=precision)
            file.write(str)
        if urdfVisual.geom_type == p.GEOM_CAPSULE:
            str = '\t\t\t\t<capsule length=\"{:.{prec}f}\" radius=\"{:.{prec}f}\"/>\n'.format( \
                urdfVisual.geom_length, urdfVisual.geom_radius, prec=precision)
            file.write(str)

        file.write("\t\t\t</geometry>\n")
        str = '\t\t\t<material name=\"{}\">\n'.format(urdfVisual.material_name)
        file.write(str)
        str = '\t\t\t\t<color rgba="{:.{prec}f} {:.{prec}f} {:.{prec}f} {:.{prec}f}" />\n'.format(
            urdfVisual.material_rgba[0], \
            urdfVisual.material_rgba[1], urdfVisual.material_rgba[2], urdfVisual.material_rgba[3], prec=precision)
        file.write(str)
        file.write("\t\t\t</material>\n")
        file.write("\t\t</visual>\n")

    def writeCollisionShape(self, file, urdfCollision, precision=5):
        file.write("\t\t<collision>\n")
        str = '\t\t\t<origin rpy="{:.{prec}f} {:.{prec}f} {:.{prec}f}" xyz="{:.{prec}f} {:.{prec}f} {:.{prec}f}"/>\n'.format( \
            urdfCollision.origin_rpy[0], urdfCollision.origin_rpy[1], urdfCollision.origin_rpy[2],
            urdfCollision.origin_xyz[0], urdfCollision.origin_xyz[1], urdfCollision.origin_xyz[2], prec=precision)
        file.write(str)
        file.write("\t\t\t<geometry>\n")
        if urdfCollision.geom_type == p.GEOM_BOX:
            str = '\t\t\t\t<box size=\"{:.{prec}f} {:.{prec}f} {:.{prec}f}\"/>\n'.format(urdfCollision.geom_extents[0], \
                                                                                         urdfCollision.geom_extents[1],
                                                                                         urdfCollision.geom_extents[2],
                                                                                         prec=precision)
            file.write(str)
        if urdfCollision.geom_type == p.GEOM_SPHERE:
            str = '\t\t\t\t<sphere radius=\"{:.{prec}f}\"/>\n'.format(urdfCollision.geom_radius, \
                                                                      prec=precision)
            file.write(str)
        if urdfCollision.geom_type == p.GEOM_MESH:
            str = '\t\t\t\t<mesh filename=\"{}\"/>\n'.format(urdfCollision.geom_meshfilename, \
                                                             prec=precision)
            file.write(str)
        if urdfCollision.geom_type == p.GEOM_CYLINDER:
            str = '\t\t\t\t<cylinder length=\"{:.{prec}f}\" radius=\"{:.{prec}f}\"/>\n'.format( \
                urdfCollision.geom_length, urdfCollision.geom_radius, prec=precision)
            file.write(str)
        if urdfCollision.geom_type == p.GEOM_CAPSULE:
            str = '\t\t\t\t<capsule length=\"{:.{prec}f}\" radius=\"{:.{prec}f}\"/>\n'.format( \
                urdfCollision.geom_length, urdfCollision.geom_radius, prec=precision)
            file.write(str)
        file.write("\t\t\t</geometry>\n")
        file.write("\t\t</collision>\n")

    def writeLink(self, file, urdfLink):
        file.write("\t<link name=\"")
        file.write(urdfLink.link_name)
        file.write("\">\n")

        self.writeInertial(file, urdfLink.urdf_inertial)
        for v in urdfLink.urdf_visual_shapes:
            self.writeVisualShape(file, v)
        for c in urdfLink.urdf_collision_shapes:
            self.writeCollisionShape(file, c)
        file.write("\t</link>\n")

    def writeJoint(self, file, urdfJoint, precision=5):
        jointTypeStr = "invalid"
        if urdfJoint.joint_type == p.JOINT_REVOLUTE:
            if urdfJoint.joint_upper_limit < urdfJoint.joint_lower_limit:
                jointTypeStr = "continuous"
            else:
                jointTypeStr = "revolute"
        if urdfJoint.joint_type == p.JOINT_FIXED:
            jointTypeStr = "fixed"
        if urdfJoint.joint_type == p.JOINT_PRISMATIC:
            jointTypeStr = "prismatic"
        str = '\t<joint name=\"{}\" type=\"{}\">\n'.format(urdfJoint.joint_name, jointTypeStr)
        file.write(str)
        str = '\t\t<parent link=\"{}\"/>\n'.format(urdfJoint.parent_name)
        file.write(str)
        str = '\t\t<child link=\"{}\"/>\n'.format(urdfJoint.child_name)
        file.write(str)

        if urdfJoint.joint_type == p.JOINT_PRISMATIC:
            # todo: handle limits
            lowerLimit = -0.5
            upperLimit = 0.5
            str = '<limit effort="1000.0" lower="{:.{prec}f}" upper="{:.{prec}f}" velocity="0.5"/>'.format(lowerLimit,
                                                                                                           upperLimit,
                                                                                                           prec=precision)
            file.write(str)

        file.write("\t\t<dynamics damping=\"1.0\" friction=\"0.0001\"/>\n")
        str = '\t\t<origin xyz=\"{:.{prec}f} {:.{prec}f} {:.{prec}f}\"/>\n'.format(urdfJoint.joint_origin_xyz[0], \
                                                                                   urdfJoint.joint_origin_xyz[1],
                                                                                   urdfJoint.joint_origin_xyz[2],
                                                                                   prec=precision)
        file.write(str)
        str = '\t\t<axis xyz=\"{:.{prec}f} {:.{prec}f} {:.{prec}f}\"/>\n'.format(urdfJoint.joint_axis_xyz[0], \
                                                                                 urdfJoint.joint_axis_xyz[1],
                                                                                 urdfJoint.joint_axis_xyz[2],
                                                                                 prec=precision)
        file.write(str)
        file.write("\t</joint>\n")

    def saveUrdf(self, fileName):
        file = open(fileName, "w")
        file.write("<?xml version=\"0.0\" ?>\n")
        file.write("<robot name=\"")
        file.write(self.robotName)
        file.write("\">\n")

        for link in self.urdfLinks:
            self.writeLink(file, link)

        for joint in self.urdfJoints:
            self.writeJoint(file, joint)

        file.write("</robot>\n")
        file.close()

    # def addLink(...)

    def createMultiBody(self, basePosition=[0, 0, 0], physicsClientId=0):
        # assume link[0] is base
        if (len(self.urdfLinks) == 0):
            return -1

        base = self.urdfLinks[0]

        # v.tmp_collision_shape_ids=[]
        baseMass = base.urdf_inertial.mass
        baseCollisionShapeIndex = -1
        baseShapeTypeArray = []
        baseRadiusArray = []
        baseHalfExtentsArray = []
        lengthsArray = []
        fileNameArray = []
        meshScaleArray = []
        basePositionsArray = []
        baseOrientationsArray = []

        for v in base.urdf_collision_shapes:
            #if v.geom_meshfilename == UNKNOWN_FILE:
            #    continue
            shapeType = v.geom_type
            baseShapeTypeArray.append(shapeType)
            baseHalfExtentsArray.append([0.5 * v.geom_extents[0], 0.5 * v.geom_extents[1], 0.5 * v.geom_extents[2]])
            baseRadiusArray.append(v.geom_radius)
            lengthsArray.append(v.geom_length)
            fileNameArray.append(v.geom_meshfilename)
            meshScaleArray.append(v.geom_meshscale)
            basePositionsArray.append(v.origin_xyz)
            baseOrientationsArray.append(p.getQuaternionFromEuler(v.origin_rpy))

        if (len(baseShapeTypeArray)):
            baseCollisionShapeIndex = p.createCollisionShapeArray(shapeTypes=baseShapeTypeArray,
                                                                  radii=baseRadiusArray,
                                                                  halfExtents=baseHalfExtentsArray,
                                                                  lengths=lengthsArray,
                                                                  fileNames=fileNameArray,
                                                                  meshScales=meshScaleArray,
                                                                  collisionFramePositions=basePositionsArray,
                                                                  collisionFrameOrientations=baseOrientationsArray,
                                                                  physicsClientId=physicsClientId)

        urdfVisuals = base.urdf_visual_shapes
        baseVisualShapeIndex = p.createVisualShapeArray(shapeTypes=[v.geom_type for v in urdfVisuals],
                                                        halfExtents=[[ext * 0.5 for ext in v.geom_extents] for v in
                                                                     urdfVisuals],
                                                        radii=[v.geom_radius for v in urdfVisuals],
                                                        #lengths=[v.geom_length[0] for v in urdfVisuals],
                                                        lengths=[v.geom_length for v in urdfVisuals],
                                                        fileNames=[v.geom_meshfilename for v in urdfVisuals],
                                                        meshScales=[v.geom_meshscale for v in urdfVisuals],
                                                        rgbaColors=[v.material_rgba for v in urdfVisuals],
                                                        visualFramePositions=[v.origin_xyz for v in urdfVisuals],
                                                        visualFrameOrientations=[p.getQuaternionFromEuler(v.origin_rpy) for v in urdfVisuals],
                                                        physicsClientId=physicsClientId)

        linkMasses = []
        linkCollisionShapeIndices = []
        linkVisualShapeIndices = []
        linkPositions = []
        linkOrientations = []
        linkMeshScaleArray = []
        linkInertialFramePositions = []
        linkInertialFrameOrientations = []
        linkParentIndices = []
        linkJointTypes = []
        linkJointAxis = []

        for joint in self.urdfJoints:
            link = joint.link
            linkMass = link.urdf_inertial.mass
            linkCollisionShapeIndex = -1
            linkVisualShapeIndex = -1
            linkParentIndex = self.linkNameToIndex[joint.parent_name]
            linkShapeTypeArray = []
            linkRadiusArray = []
            linkHalfExtentsArray = []
            lengthsArray = []
            fileNameArray = []
            linkPositionsArray = []
            linkOrientationsArray = []

            for v in link.urdf_collision_shapes:
                #if v.geom_meshfilename == UNKNOWN_FILE:
                #    continue
                shapeType = v.geom_type
                linkShapeTypeArray.append(shapeType)
                linkHalfExtentsArray.append([0.5 * v.geom_extents[0], 0.5 * v.geom_extents[1], 0.5 * v.geom_extents[2]])
                linkRadiusArray.append(v.geom_radius)
                lengthsArray.append(v.geom_length)
                fileNameArray.append(v.geom_meshfilename)
                linkMeshScaleArray.append(v.geom_meshscale)
                linkPositionsArray.append(v.origin_xyz)
                linkOrientationsArray.append(p.getQuaternionFromEuler(v.origin_rpy))

            if (len(linkShapeTypeArray)):
                linkCollisionShapeIndex = p.createCollisionShapeArray(shapeTypes=linkShapeTypeArray,
                                                                      radii=linkRadiusArray,
                                                                      halfExtents=linkHalfExtentsArray,
                                                                      lengths=lengthsArray,
                                                                      fileNames=fileNameArray,
                                                                      meshScales=linkMeshScaleArray,
                                                                      collisionFramePositions=linkPositionsArray,
                                                                      collisionFrameOrientations=linkOrientationsArray,
                                                                      physicsClientId=physicsClientId)

                urdfVisuals = link.urdf_visual_shapes
                linkVisualShapeIndex = p.createVisualShapeArray(shapeTypes=[v.geom_type for v in urdfVisuals],
                                                                halfExtents=[[ext * 0.5 for ext in v.geom_extents] for v
                                                                             in urdfVisuals],
                                                                radii=[v.geom_radius for v in urdfVisuals],
                                                                #lengths=[v.geom_length[0] for v in urdfVisuals],
                                                                lengths=[v.geom_length for v in urdfVisuals],
                                                                fileNames=[v.geom_meshfilename for v in urdfVisuals],
                                                                meshScales=[v.geom_meshscale for v in urdfVisuals],
                                                                rgbaColors=[v.material_rgba for v in urdfVisuals],
                                                                visualFramePositions=[v.origin_xyz for v in
                                                                                      urdfVisuals],
                                                                visualFrameOrientations=[p.getQuaternionFromEuler(v.origin_rpy) for v in
                                                                                         urdfVisuals],
                                                                physicsClientId=physicsClientId)

            linkMasses.append(linkMass)
            linkCollisionShapeIndices.append(linkCollisionShapeIndex)
            linkVisualShapeIndices.append(linkVisualShapeIndex)
            linkPositions.append(joint.joint_origin_xyz)
            linkOrientations.append(p.getQuaternionFromEuler(joint.joint_origin_rpy))
            linkInertialFramePositions.append(link.urdf_inertial.origin_xyz)
            linkInertialFrameOrientations.append(p.getQuaternionFromEuler(link.urdf_inertial.origin_rpy))
            linkParentIndices.append(linkParentIndex)
            linkJointTypes.append(joint.joint_type)
            linkJointAxis.append(joint.joint_axis_xyz)
        obUid = p.createMultiBody(baseMass, \
                                  baseCollisionShapeIndex=baseCollisionShapeIndex,
                                  baseVisualShapeIndex=baseVisualShapeIndex,
                                  basePosition=basePosition,
                                  baseInertialFramePosition=base.urdf_inertial.origin_xyz,
                                  baseInertialFrameOrientation=p.getQuaternionFromEuler(base.urdf_inertial.origin_rpy),
                                  linkMasses=linkMasses,
                                  linkCollisionShapeIndices=linkCollisionShapeIndices,
                                  linkVisualShapeIndices=linkVisualShapeIndices,
                                  linkPositions=linkPositions,
                                  linkOrientations=linkOrientations,
                                  linkInertialFramePositions=linkInertialFramePositions,
                                  linkInertialFrameOrientations=linkInertialFrameOrientations,
                                  linkParentIndices=linkParentIndices,
                                  linkJointTypes=linkJointTypes,
                                  linkJointAxis=linkJointAxis,
                                  physicsClientId=physicsClientId)
        return obUid

    def __del__(self):
        pass
