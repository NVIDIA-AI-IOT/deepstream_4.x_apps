/*
 * Copyright (c) 2019 NVIDIA Corporation.  All rights reserved.
 *
 * NVIDIA Corporation and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA Corporation is strictly prohibited.
 *
 */

#include <gst/gst.h>
#include <glib.h>
#include <stdio.h>
#include "gstnvdsmeta.h"

/* The muxer output resolution must be set if the input streams will be of
 * different resolution. The muxer will scale all the input frames to this
 * resolution. */
#define MUXER_OUTPUT_WIDTH 1920
#define MUXER_OUTPUT_HEIGHT 1080

/* Muxer batch formation timeout, for e.g. 40 millisec. Should ideally be set
 * based on the fastest source's framerate. */
#define MUXER_BATCH_TIMEOUT_USEC 4000000

static gboolean
bus_call (GstBus * bus, GstMessage * msg, gpointer data) {
    GMainLoop *loop = (GMainLoop *) data;
    switch (GST_MESSAGE_TYPE (msg)) {
    case GST_MESSAGE_EOS:
        g_print ("End of stream\n");
        g_main_loop_quit (loop);
        break;
    case GST_MESSAGE_ERROR: {
        gchar *debug;
        GError *error;
        gst_message_parse_error (msg, &error, &debug);
        g_printerr ("ERROR from element %s: %s\n",
                    GST_OBJECT_NAME (msg->src), error->message);
        if (debug)
            g_printerr ("Error details: %s\n", debug);
        g_free (debug);
        g_error_free (error);
        g_main_loop_quit (loop);
        break;
    }
    default:
        break;
    }
    return TRUE;
}

int
main (int argc, char *argv[]) {
    GMainLoop *loop = NULL;
    GstElement *pipeline = NULL, *source = NULL, *h264parser = NULL,
               *decoder = NULL, *streammux = NULL, *sink = NULL,
               *pgie = NULL, *nvvidconv = NULL, *nvdsosd = NULL;
#ifdef PLATFORM_TEGRA
    GstElement *transform = NULL;
#endif
    GstBus *bus = NULL;
    guint bus_watch_id;
    GstPad *osd_sink_pad = NULL;

    /* Check input arguments */
    if (argc != 3) {
        g_printerr ("Usage: %s config_file <H264 filename>\n", argv[0]);
        return -1;
    }

    /* Standard GStreamer initialization */
    gst_init (&argc, &argv);
    loop = g_main_loop_new (NULL, FALSE);

    /* Create gstreamer elements */
    /* Create Pipeline element that will form a connection of other elements */
    pipeline = gst_pipeline_new ("ds-custom-pipeline");

    /* Source element for reading from the file */
    source = gst_element_factory_make ("filesrc", "file-source");

    /* Since the data format in the input file is elementary h264 stream,
     * we need a h264parser */
    h264parser = gst_element_factory_make ("h264parse", "h264-parser");

    /* Use nvdec_h264 for hardware accelerated decode on GPU */
    decoder = gst_element_factory_make ("nvv4l2decoder", "nvv4l2-decoder");
    g_object_set (G_OBJECT (decoder), "bufapi-version", 1, NULL);

    /* Create nvstreammux instance to form batches from one or more sources. */
    streammux = gst_element_factory_make ("nvstreammux", "stream-muxer");

    if (!pipeline || !streammux) {
        g_printerr ("One element could not be created. Exiting.\n");
        return -1;
    }

    /* Use nvinfer to run inferencing on decoder's output,
     * behaviour of inferencing is set through config file */
    pgie = gst_element_factory_make ("nvinfer", "primary-nvinference-engine");

    /* Use convertor to convert from NV12 to RGBA as required by nvdsosd */
    nvvidconv = gst_element_factory_make ("nvvideoconvert", "nvvideo-converter");

    /* Create OSD to draw on the converted RGBA buffer */
    nvdsosd = gst_element_factory_make ("nvdsosd", "nv-onscreendisplay");

    /* Finally render the osd output */
#ifdef PLATFORM_TEGRA
    transform = gst_element_factory_make ("nvegltransform", "nvegl-transform");
#endif
    sink = gst_element_factory_make ("nveglglessink", "nvvideo-renderer");

    if (!source || !h264parser || !decoder || !pgie
            || !nvvidconv || !nvdsosd || !sink) {
        g_printerr ("One element could not be created. Exiting.\n");
        return -1;
    }

#ifdef PLATFORM_TEGRA
    if(!transform) {
        g_printerr ("One tegra element could not be created. Exiting.\n");
        return -1;
    }
#endif

    /* we set the input filename to the source element */
    g_object_set (G_OBJECT (source), "location", argv[2], NULL);

    g_object_set (G_OBJECT (streammux), "width", MUXER_OUTPUT_WIDTH, "height",
                  MUXER_OUTPUT_HEIGHT, "batch-size", 1,
                  "batched-push-timeout", MUXER_BATCH_TIMEOUT_USEC, NULL);

    /* Set all the necessary properties of the nvinfer element,
     * the necessary ones are : */
    g_object_set (G_OBJECT (pgie),
                  "config-file-path", argv[1], NULL);

    /* we add a message handler */
    bus = gst_pipeline_get_bus (GST_PIPELINE (pipeline));
    bus_watch_id = gst_bus_add_watch (bus, bus_call, loop);
    gst_object_unref (bus);

    /* Set up the pipeline */
    /* we add all elements into the pipeline */
#ifdef PLATFORM_TEGRA
    gst_bin_add_many (GST_BIN (pipeline),
                      source, h264parser, decoder, streammux, pgie,
                      nvvidconv, nvdsosd, transform, sink, NULL);
#else
    gst_bin_add_many (GST_BIN (pipeline),
                      source, h264parser, decoder, streammux, pgie,
                      nvvidconv, nvdsosd, sink, NULL);
#endif

    GstPad *sinkpad, *srcpad;
    gchar pad_name_sink[16] = "sink_0";
    gchar pad_name_src[16] = "src";

    sinkpad = gst_element_get_request_pad (streammux, pad_name_sink);
    if (!sinkpad) {
        g_printerr ("Streammux request sink pad failed. Exiting.\n");
        return -1;
    }

    srcpad = gst_element_get_static_pad (decoder, pad_name_src);
    if (!srcpad) {
        g_printerr ("Decoder request src pad failed. Exiting.\n");
        return -1;
    }

    if (gst_pad_link (srcpad, sinkpad) != GST_PAD_LINK_OK) {
        g_printerr ("Failed to link decoder to stream muxer. Exiting.\n");
        return -1;
    }

    gst_object_unref (sinkpad);
    gst_object_unref (srcpad);

    /* We link the elements together */
    /* file-source -> h264-parser -> nvv4l2decoder ->
     * nvinfer -> nvvideoconvert -> nvdsosd -> video-renderer */

    if (!gst_element_link_many (source, h264parser, decoder, NULL)) {
        g_printerr ("Elements could not be linked: 1. Exiting.\n");
        return -1;
    }

#ifdef PLATFORM_TEGRA
    if (!gst_element_link_many (streammux, pgie,
                                nvvidconv, nvdsosd, transform, sink, NULL)) {
        g_printerr ("Elements could not be linked: 2. Exiting.\n");
        return -1;
    }
#else
    if (!gst_element_link_many (streammux, pgie,
                                nvvidconv, nvdsosd, sink, NULL)) {
        g_printerr ("Elements could not be linked: 2. Exiting.\n");
        return -1;
    }
#endif

    /* Set the pipeline to "playing" state */
    g_print ("Now playing: %s\n", argv[1]);
    gst_element_set_state (pipeline, GST_STATE_PLAYING);

    /* Wait till pipeline encounters an error or EOS */
    g_print ("Running...\n");
    g_main_loop_run (loop);

    /* Out of the main loop, clean up nicely */
    g_print ("Returned, stopping playback\n");
    gst_element_set_state (pipeline, GST_STATE_NULL);
    g_print ("Deleting pipeline\n");
    gst_object_unref (GST_OBJECT (pipeline));
    g_source_remove (bus_watch_id);
    g_main_loop_unref (loop);
    return 0;
}
