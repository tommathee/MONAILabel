/*
Copyright (c) MONAI Consortium
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package qupath.lib.extension.monailabel.commands;

import java.awt.*;
import java.awt.event.KeyEvent;
import java.io.FileOutputStream;

import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.nio.file.Path;
import java.util.*;
import java.util.List;

import javax.xml.parsers.ParserConfigurationException;
import javax.xml.parsers.DocumentBuilderFactory;
import javax.xml.transform.TransformerException;
import javax.xml.parsers.DocumentBuilder;
import javax.xml.transform.TransformerFactory;
import javax.xml.transform.dom.DOMSource;
import javax.xml.transform.stream.StreamResult;
import javax.xml.transform.Transformer;
import javax.xml.transform.OutputKeys;

import javafx.application.Platform;
import javafx.concurrent.Task;
import javafx.scene.Scene;
import javafx.scene.control.ProgressIndicator;
import javafx.scene.input.KeyCode;

import javafx.scene.layout.StackPane;

import javafx.stage.Modality;
import javafx.stage.Stage;
import org.controlsfx.dialog.ProgressDialog;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.w3c.dom.Document;
import org.w3c.dom.Node;
import org.w3c.dom.NodeList;
import org.xml.sax.SAXException;
import org.w3c.dom.Element;

import qupath.lib.extension.monailabel.MonaiLabelClient;
import qupath.lib.extension.monailabel.MonaiLabelClient.RequestInfer;
import qupath.lib.extension.monailabel.MonaiLabelClient.ResponseInfo;
import qupath.lib.extension.monailabel.Settings;
import qupath.lib.extension.monailabel.MonaiLabelClient.ImageInfo;
import qupath.lib.extension.monailabel.Utils;
import qupath.lib.geom.Point2;
import qupath.lib.gui.QuPathGUI;
import qupath.lib.gui.dialogs.Dialogs;
import qupath.lib.images.ImageData;
import qupath.lib.images.writers.ImageWriterTools;
import qupath.lib.objects.PathObject;
import qupath.lib.objects.PathObjects;
import qupath.lib.objects.PathROIObject;
import qupath.lib.objects.classes.PathClass;
import qupath.lib.objects.classes.PathClassFactory;
import qupath.lib.plugins.parameters.ParameterList;
import qupath.lib.regions.ImagePlane;
import qupath.lib.regions.RegionRequest;
import qupath.lib.roi.PointsROI;
import qupath.lib.roi.ROIs;
import qupath.lib.roi.RectangleROI;
import qupath.lib.roi.interfaces.ROI;
import qupath.lib.scripting.QP;
import qupath.lib.objects.PathCellObject;

public class RunInference implements Runnable {
  private final static Logger logger = LoggerFactory.getLogger(RunInference.class);

  private QuPathGUI qupath;
  private static String selectedModel;
  private static int[] selectedBBox;
  private static int selectedTileSize = 1024;

  public RunInference(QuPathGUI qupath) {
    this.qupath = qupath;
  }

  @Override
  public void run() {
    Path annotationXML = null;
    //addUndoConfirmationListener(qupath.getStage());
    try {
      var viewer = qupath.getViewer();
      var imageData = viewer.getImageData();
      var selected = imageData.getHierarchy().getSelectionModel().getSelectedObject();
      var roi = selected != null ? selected.getROI() : null;

      if (roi == null || !(roi instanceof RectangleROI)) {
        Dialogs.showPlainMessage("Please create and select ROI", "Please create and select a Rectangle ROI before " +
                "running this method.\nThe \"Annotations\" function creates annotations within the rectangle.");
        return;
      }

      String imageFile = Utils.getFileName(viewer.getImageData().getServerPath());
      String image = Utils.getNameWithoutExtension(imageFile);
      String im = imageFile.toLowerCase();
      boolean isWSI = !im.endsWith(".png") && !im.endsWith(".jpg") && !im.endsWith(".jpeg");
      logger.info("MONAILabel:: isWSI: " + isWSI + "; File: " + imageFile);

      /*
      // Select first RectangleROI if not selected explicitly
      if (isWSI && (roi == null || !(roi instanceof RectangleROI))) {
        List<PathObject> objs = imageData.getHierarchy().getFlattenedObjectList(null);
        for (int i = 0; i < objs.size(); i++) {
          var obj = objs.get(i);
          ROI r = obj.getROI();

          if (r instanceof RectangleROI) {
            roi = r;
            Dialogs.showWarningNotification("MONALabel",
                "ROI is NOT explicitly selected; using first Rectangle ROI from Hierarchy");
            imageData.getHierarchy().getSelectionModel().setSelectedObject(obj);
            break;
          }
        }
      }
      */

      int[] bbox = Utils.getBBOX(roi);
      int tileSize = selectedTileSize;
      if (isWSI && bbox[2] == 0 && bbox[3] == 0 && selectedBBox != null) {
        bbox = selectedBBox;
      }

      ResponseInfo info = MonaiLabelClient.info();
      List<String> names = Arrays.asList(info.models.keySet().toArray(new String[0]));

      if (selectedModel == null || selectedModel.isEmpty()) {
        selectedModel = names.isEmpty() ? "" : names.get(0);
      }

      ParameterList list = new ParameterList();
      list.addChoiceParameter("Model", "Model Name", selectedModel, names);
      list.addTitleParameter("Parameters of selected ROI:");
      if (isWSI) {
        list.addEmptyParameter("(do not change, if not necessary)");
        list.addStringParameter("Location", "Location (x,y,w,h)", Arrays.toString(bbox));
        list.addIntParameter("TileSize", "TileSize", tileSize);
        annotationXML = getAnnotationsXml(image, imageData, new int[4]);
        boolean validImage = MonaiLabelClient.imageExists(image);
        if (!validImage) {
          Path imagePatch = java.nio.file.Files.createTempFile("patch", ".png");
          String patchName = isWSI ? image + String.format("-patch-%d_%d_%d_%d", bbox[0], bbox[1], bbox[2], bbox[3])
              : image;
          ImageInfo imageInfo = MonaiLabelClient.saveImage(patchName, imagePatch.toFile(), "{}");
          MonaiLabelClient.saveLabel(imageInfo.image, annotationXML.toFile(), null, "{}");
        }
        if (validImage) {
          MonaiLabelClient.saveLabel(image, annotationXML.toFile(), null, "{}");
        }
      }

      if (Dialogs.showParameterDialog("MONAILabel", list)) {
        String model = (String) list.getChoiceParameterValue("Model");
        if (isWSI) {
          bbox = Utils.parseStringArray(list.getStringParameterValue("Location"));
          tileSize = list.getIntParameterValue("TileSize").intValue();
        } else {
          bbox = new int[] { 0, 0, 0, 0 };
          tileSize = selectedTileSize;
        }

        selectedModel = model;
        selectedBBox = bbox;
        selectedTileSize = tileSize;

        final int[] finalBbox = bbox;
        final int finalTileSize = tileSize;

        // running inference and progress dialog in threads
        Task<Void> task = new Task<Void>() {
          @Override
          protected Void call() throws Exception {
            runInference(model, info, finalBbox, finalTileSize, imageData, imageFile, isWSI);
            return null;
          }
        };

        ProgressDialog progressDialog = new ProgressDialog(task);
        progressDialog.setTitle("MONAILabel");
        progressDialog.setHeaderText("Server-side processing is in progress...");
        progressDialog.initOwner(qupath.getStage());

        // Start the task
        new Thread(task).start();

        // Wait for the task to finish
        task.setOnSucceeded(event -> {
          progressDialog.close();
          // Autosave project
          try {
            Robot robot = new Robot();
            robot.keyPress(KeyEvent.VK_CONTROL);
            robot.keyPress(KeyEvent.VK_S);
            robot.keyRelease(KeyEvent.VK_S);
            robot.keyRelease(KeyEvent.VK_CONTROL);
            Dialogs.showInfoNotification("Project Autosave", "Project has been automatically saved.");
          } catch (Exception e) {
            Dialogs.showErrorMessage("Project Autosave", "Error occurred while autosaving the project.");
          }
        });
        task.setOnFailed(event -> {
          progressDialog.close();
          Throwable ex = task.getException();
          if (ex != null) {
            ex.printStackTrace();
            Dialogs.showErrorMessage("MONAILabel", ex);
          }
        });
      }

      imageData.getHierarchy().removeObject(imageData.getHierarchy().getSelectionModel().getSelectedObject(), true);
      imageData.getHierarchy().getSelectionModel().clearSelection();

    } catch (Exception ex) {
      ex.printStackTrace();
      Dialogs.showErrorMessage("MONAILabel", ex);
    }
  }

  private Path getAnnotationsXml(String image, ImageData<BufferedImage> imageData, int[] bbox)
      throws IOException, ParserConfigurationException, TransformerException {
    DocumentBuilderFactory docFactory = DocumentBuilderFactory.newInstance();
    DocumentBuilder docBuilder = docFactory.newDocumentBuilder();

    // root elements
    Document doc = docBuilder.newDocument();
    Element rootElement = doc.createElement("ASAP_Annotations");
    doc.appendChild(rootElement);

    Element annotations = doc.createElement("Annotations");
    annotations.setAttribute("Name", "");
    annotations.setAttribute("Description", "");
    annotations.setAttribute("X", String.valueOf(bbox[0]));
    annotations.setAttribute("Y", String.valueOf(bbox[1]));
    annotations.setAttribute("W", String.valueOf(bbox[2]));
    annotations.setAttribute("H", String.valueOf(bbox[3]));
    rootElement.appendChild(annotations);

    ROI patchROI = (bbox[2] > 0 && bbox[3] > 0) ? ROIs.createRectangleROI(bbox[0], bbox[1], bbox[2], bbox[3], null)
        : null;

    int count = 0;
    var groups = new HashMap<String, String>();
    List<PathObject> objs = imageData.getHierarchy().getFlattenedObjectList(null);
    for (int i = 0; i < objs.size(); i++) {
      var a = objs.get(i);

      // Ignore which doesn't have class
      String name = a.getPathClass() != null ? a.getPathClass().getName() : null;
      if (name == null || name.isEmpty()) {
        continue;
      }

      var roi = a.getROI();
      if (a.isCell()) {
        roi = ((PathCellObject) a).getNucleusROI();
      }

      // Ignore Points
      if (roi == null || roi.isPoint()) {
        continue;
      }

      // Ignore other objects not part of BBOX
      if (patchROI != null && !patchROI.contains(roi.getCentroidX(), roi.getCentroidY())) {
        continue;
      }

      var points = roi.getAllPoints();
      var color = String.format("#%06x", 0xFFFFFF & a.getPathClass().getColor());
      groups.put(name, color);

      Element annotation = doc.createElement("Annotation");
      annotation.setAttribute("Name", name);
      annotation.setAttribute("Type", roi.getRoiName());
      annotation.setAttribute("PartOfGroup", name);
      annotation.setAttribute("Color", color);
      annotations.appendChild(annotation);

      Element coordinates = doc.createElement("Coordinates");
      annotation.appendChild(coordinates);

      for (int j = 0; j < points.size(); j++) {
        var p = points.get(j);
        Element coordinate = doc.createElement("Coordinate");
        coordinate.setAttribute("Order", String.valueOf(j));
        coordinate.setAttribute("X", String.valueOf((int) p.getX() - bbox[0]));
        coordinate.setAttribute("Y", String.valueOf((int) p.getY() - bbox[1]));
        coordinates.appendChild(coordinate);
      }
      count++;
    }

    Element annotationGroups = doc.createElement("AnnotationGroups");
    rootElement.appendChild(annotationGroups);

    for (String group : groups.keySet()) {
      Element annotationGroup = doc.createElement("Group");
      annotationGroup.setAttribute("Name", group);
      annotationGroup.setAttribute("PartOfGroup", "None");
      annotationGroup.setAttribute("Color", groups.get(group));
      annotationGroups.appendChild(annotationGroup);
    }

    logger.info("Total Objects saved: " + count);
    if (count == 0) {
      throw new IOException("ZERO annotations found (nothing to save/submit)");
    }
    return writeXml(image, doc);
  }

  private Path writeXml(String image, Document doc) throws TransformerException, IOException {
    FileOutputStream output = null;
    try {
      var path = java.nio.file.Files.createTempFile(image, ".xml");
      output = new FileOutputStream(path.toFile());

      TransformerFactory transformerFactory = TransformerFactory.newInstance();
      Transformer transformer = transformerFactory.newTransformer();

      // pretty print
      transformer.setOutputProperty(OutputKeys.INDENT, "yes");

      DOMSource source = new DOMSource(doc);
      StreamResult result = new StreamResult(output);
      transformer.transform(source, result);
      output.close();
      return path;
    } finally {
      if (output != null) {
        output.close();
      }
    }
  }

  public static ArrayList<Point2> getClicks(String name, ImageData<BufferedImage> imageData, ROI monaiLabelROI,
      int offsetX, int offsetY) {
    List<PathObject> objs = imageData.getHierarchy().getFlattenedObjectList(null);
    ArrayList<Point2> clicks = new ArrayList<Point2>();
    for (int i = 0; i < objs.size(); i++) {
      var obj = objs.get(i);
      String pname = obj.getPathClass() == null ? "" : obj.getPathClass().getName();
      if (name.isEmpty() || pname.equalsIgnoreCase(name)) {
        ROI r = obj.getROI();
        if (r instanceof PointsROI) {
          List<Point2> points = r.getAllPoints();
          for (Point2 p : points) {
            if (monaiLabelROI.contains(p.getX(), p.getY())) {
              clicks.add(new Point2(p.getX() - offsetX, p.getY() - offsetY));
            }
          }
        }
      }
    }

    logger.info("MONAILabel:: Total " + name + " clicks/points: " + clicks.size());
    return clicks;
  }

  public static void runInference(String model, ResponseInfo info, int[] bbox, int tileSize,
      ImageData<BufferedImage> imageData, String imageFile, boolean isWSI)
      throws SAXException, IOException, ParserConfigurationException, InterruptedException {
    logger.info("MONAILabel:: Running Inference...; model = " + model);

    boolean isNuClick = info.models.get(model).nuclick;
    boolean override = !isNuClick;
    boolean validateClicks = isNuClick;
    var labels = new HashSet<String>(Arrays.asList(info.models.get(model).labels.labels()));

    logger.info("MONAILabel:: Model: " + model + "; Labels: " + labels);

    Path imagePatch = null;
    try {
      RequestInfer req = new RequestInfer();
      req.location[0] = bbox[0];
      req.location[1] = bbox[1];
      req.size[0] = bbox[2];
      req.size[1] = bbox[3];
      req.tile_size[0] = tileSize;
      req.tile_size[1] = tileSize;

      ROI roi = ROIs.createRectangleROI(bbox[0], bbox[1], bbox[2], bbox[3], null);

      String image = Utils.getNameWithoutExtension(imageFile);
      req.image_name = image;
      String sessionId = null;
      int offsetX = 0;
      int offsetY = 0;

      // check if image exists on server
      if (!MonaiLabelClient.imageExists(image) && (sessionId == null || sessionId.isEmpty())) {
        logger.info("MONAILabel:: Image does not exist on Server.");

        image = null;
        offsetX = req.location[0];
        offsetY = req.location[1];

        // req.location[0] = req.location[1] = 0;
        // req.size[0] = req.size[1] = 0;

        String im = imageFile.toLowerCase();
        if ((im.endsWith(".png") || im.endsWith(".jpg") || im.endsWith(".jpeg"))
            && new File(imageFile).exists()) {
          logger.info("Simple Image.. will directly upload the same");
          offsetX = offsetY = 0;
          Dialogs.showWarningNotification("MONAILabel",
              "Ignoring ROI; Running Inference over full non-wsi Image");
        } else {
          if (bbox[2] == 0 && bbox[3] == 0) {
            Dialogs.showErrorMessage("MONAILabel",
                "Can not run WSI Inference on a remote image (Not exists in Datastore)");
            return;
          }

          imagePatch = java.nio.file.Files.createTempFile("patch", ".png");
          imageFile = imagePatch.toString();
          var requestROI = RegionRequest.createInstance(imageData.getServer().getPath(), 1, roi);
          ImageWriterTools.writeImageRegion(imageData.getServer(), requestROI, imageFile);
        }
      }

      ArrayList<Point2> fg = new ArrayList<>();
      ArrayList<Point2> bg = new ArrayList<>();
      if (isNuClick) {
        fg = getClicks("", imageData, roi, offsetX, offsetY);
      } else {
        fg = getClicks("Positive", imageData, roi, offsetX, offsetY);
        bg = getClicks("Negative", imageData, roi, offsetX, offsetY);
      }

      if (validateClicks) {
        if (fg.size() == 0 && bg.size() == 0) {
          Dialogs.showErrorMessage("MONAILabel",
              "Need atleast one Postive/Negative annotation/click point within the ROI");
          return;
        }
        if (roi.getBoundsHeight() < 128 || roi.getBoundsWidth() < 128) {
          Dialogs.showErrorMessage("MONAILabel", "Min Height/Width of ROI should be more than 128");
          return;
        }
      }
      req.params.addClicks(fg, true);
      req.params.addClicks(bg, false);
      req.params.max_workers = Settings.maxWorkersProperty().intValue();

      Document dom = MonaiLabelClient.infer(model, image, imageFile, sessionId, req);
      NodeList annotation_list = dom.getElementsByTagName("Annotation");
      int count = updateAnnotations(labels, annotation_list, roi, imageData, override, offsetX, offsetY);

      // Update hierarchy to see changes in QuPath's hierarchy
      QP.fireHierarchyUpdate(imageData.getHierarchy());
      logger.info("MONAILabel:: Annotation Done! => Total Objects Added: " + count);
    } finally {
      Utils.deleteFile(imagePatch);
    }
  }

  public static int updateAnnotations(Set<String> labels, NodeList annotation_list, ROI roi,
      ImageData<BufferedImage> imageData, boolean override, int offsetX, int offsetY) {
    if (override) {
      List<PathObject> objs = imageData.getHierarchy().getFlattenedObjectList(null);
      for (int i = 0; i < objs.size(); i++) {
        String name = objs.get(i).getPathClass() != null ? objs.get(i).getPathClass().getName() : null;
        if (name != null && labels.contains(name)) {
          ROI r = objs.get(i).getROI();
          if (roi.contains(r.getCentroidX(), r.getCentroidY())) {
            imageData.getHierarchy().removeObjectWithoutUpdate(objs.get(i), false);
          }
        }
      }
    } else {
      List<PathObject> objs = imageData.getHierarchy().getFlattenedObjectList(null);
      for (int i = 0; i < objs.size(); i++) {
        var obj = objs.get(i);
        ROI r = obj.getROI();
        if (r instanceof PointsROI) {
          String pname = obj.getPathClass() == null ? "" : obj.getPathClass().getName();
          if (pname.equalsIgnoreCase("Positive") || pname.equalsIgnoreCase("Negative")) {
            continue;
          }
          imageData.getHierarchy().removeObjectWithoutUpdate(obj, false);
        }
      }
      QP.fireHierarchyUpdate(imageData.getHierarchy());
    }

    int count = 0;
    for (int i = 0; i < annotation_list.getLength(); i++) {
      Node annotation = annotation_list.item(i);
      String annotationClass = annotation.getAttributes().getNamedItem("Name").getTextContent();
      // logger.info("Annotation Class: " + annotationClass);

      NodeList coordinates_list = annotation.getChildNodes();
      for (int j = 0; j < coordinates_list.getLength(); j++) {
        Node coordinates = coordinates_list.item(j);
        if (coordinates.getNodeType() != Node.ELEMENT_NODE) {
          continue;
        }

        NodeList coordinate_list = coordinates.getChildNodes();
        // logger.info("Total Coordinate: " + coordinate_list.getLength());

        ArrayList<Point2> pointsList = new ArrayList<>();
        for (int k = 0; k < coordinate_list.getLength(); k++) {
          Node coordinate = coordinate_list.item(k);
          if (coordinate.getAttributes() != null) {
            double px = offsetX
                + Double.parseDouble(coordinate.getAttributes().getNamedItem("X").getTextContent());
            double py = offsetY
                + Double.parseDouble(coordinate.getAttributes().getNamedItem("Y").getTextContent());
            pointsList.add(new Point2(px, py));
          }
        }
        if (pointsList.isEmpty()) {
          continue;
        }

        ImagePlane plane = ImagePlane.getPlane(0, 0);
        ROI polyROI = ROIs.createPolygonROI(pointsList, plane);
        PathObject annotationObject = PathObjects.createAnnotationObject(polyROI);

        PathClass pclass = PathClassFactory.getPathClass(annotationClass, Color.RED.getRGB());
        annotationObject.setPathClass(pclass);

        imageData.getHierarchy().addPathObjectWithoutUpdate(annotationObject);
        count++;
      }
    }

    return count;
  }

  /*
  private void addUndoConfirmationListener(Stage stage) {
    stage.addEventFilter(KeyEvent.KEY_PRESSED, event -> {
      if (event.isControlDown() && event.getCode() == KeyCode.Z || event.isMetaDown() && event.getCode() == KeyCode.Z) {
        event.consume(); // Prevent the original event from being processed further

        // Show a confirmation dialog
        boolean shouldUndo = Dialogs.showYesNoDialog("Undo Confirmation",
                "Are you sure you want to undo the last action? This might delete or modify your annotations.");
        if (shouldUndo) {
          // If the user confirms, perform the undo action
          qupath.getUndoRedoManager().undoOnce();
        }
      }
    });
  }

   */
}
