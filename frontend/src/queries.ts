import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import * as api from "./api";
import type { AppSettings, Episode, Export } from "./types";

/** Query keys + data hooks — the single source of server state. Components read these
 *  instead of fetching/polling by hand; mutations invalidate the right keys so the UI
 *  stays consistent without manual reloads. */
export const keys = {
  providers: ["providers"] as const,
  appSettings: ["app-settings"] as const,
  models: (connection: string) => ["app-settings", "models", connection] as const,
  voices: (connection: string, type = "") => ["app-settings", "voices", connection, type] as const,
  seriesList: (q: string) => ["series", { q }] as const,
  series: (id: string) => ["series", id] as const,
  episodes: (id: string) => ["series", id, "episodes"] as const,
  exports: (id: string) => ["series", id, "exports"] as const,
};

// ----- queries --------------------------------------------------------------
export const useProviders = () => useQuery({ queryKey: keys.providers, queryFn: api.loadProviders });

export const useAppSettings = () => useQuery({ queryKey: keys.appSettings, queryFn: api.loadAppSettings });

export const useConnectionModels = (connection: string) =>
  useQuery({ queryKey: keys.models(connection), queryFn: () => api.listConnectionModels(connection), enabled: !!connection });

export const useTtsVoices = (connection: string, type = "") =>
  useQuery({ queryKey: keys.voices(connection, type), queryFn: () => api.listTtsVoices(connection, type), enabled: !!connection, retry: false });

export const useSeriesList = (q: string) =>
  useQuery({ queryKey: keys.seriesList(q), queryFn: () => api.listSeries(q) });

export const useSeries = (id: string) =>
  useQuery({ queryKey: keys.series(id), queryFn: () => api.getSeries(id), enabled: !!id });

export const useEpisodes = (id: string) =>
  useQuery({
    queryKey: keys.episodes(id),
    queryFn: () => api.listEpisodes(id),
    enabled: !!id,
    // Poll only while a dub is in flight, then stop.
    refetchInterval: (query) =>
      (query.state.data as Episode[] | undefined)?.some((e) => e.dub_state === "translating") ? 4000 : false,
  });

export const useExports = (id: string) =>
  useQuery({
    queryKey: keys.exports(id),
    queryFn: () => api.listExports(id),
    enabled: !!id,
    refetchInterval: (query) =>
      (query.state.data as Export[] | undefined)?.some((x) => x.state === "rendering") ? 4000 : false,
  });

// ----- mutations ------------------------------------------------------------
/** Invalidate everything under a series (detail counts + episodes + exports). */
export function useSeriesInvalidator() {
  const qc = useQueryClient();
  return (seriesId?: string) => {
    qc.invalidateQueries({ queryKey: ["series"] });
    if (seriesId) {
      qc.invalidateQueries({ queryKey: keys.episodes(seriesId) });
      qc.invalidateQueries({ queryKey: keys.exports(seriesId) });
    }
  };
}

export function useScanSeries(seriesId: string) {
  const invalidate = useSeriesInvalidator();
  return useMutation({ mutationFn: () => api.scanSeries(seriesId), onSuccess: () => invalidate(seriesId) });
}

export function useAddSource(seriesId: string) {
  const invalidate = useSeriesInvalidator();
  return useMutation({ mutationFn: (url: string) => api.addSource(seriesId, "douyin", url), onSuccess: () => invalidate(seriesId) });
}

export function useDeleteSource(seriesId: string) {
  const invalidate = useSeriesInvalidator();
  return useMutation({ mutationFn: (sourceId: string) => api.deleteSource(seriesId, sourceId), onSuccess: () => invalidate(seriesId) });
}

export function useDeleteSeries() {
  const invalidate = useSeriesInvalidator();
  return useMutation({ mutationFn: (seriesId: string) => api.deleteSeries(seriesId), onSuccess: () => invalidate() });
}

export function useTranslate(seriesId: string) {
  const invalidate = useSeriesInvalidator();
  return useMutation({ mutationFn: (episodeIds: string[]) => api.translateEpisodes(seriesId, episodeIds), onSuccess: () => invalidate(seriesId) });
}

export function useDeleteEpisodeDub(seriesId: string) {
  const invalidate = useSeriesInvalidator();
  return useMutation({ mutationFn: (episodeId: string) => api.deleteEpisodeDub(episodeId), onSuccess: () => invalidate(seriesId) });
}

export function useTranslateTitles(seriesId: string) {
  const invalidate = useSeriesInvalidator();
  return useMutation({ mutationFn: () => api.translateTitles(seriesId), onSuccess: () => invalidate(seriesId) });
}

export function useCreateExport(seriesId: string) {
  const invalidate = useSeriesInvalidator();
  return useMutation({
    mutationFn: async (body: { title: string; description: string; hashtags: string[]; episode_job_ids: string[] }) => {
      const created = await api.createExport(seriesId, body);
      await api.renderExport(created.id);
      return created;
    },
    onSuccess: () => invalidate(seriesId),
  });
}

export function useSuggestExport(seriesId: string) {
  return useMutation({ mutationFn: (episodeIds: string[]) => api.suggestExport(seriesId, episodeIds) });
}

export function useRenderExport(seriesId: string) {
  const invalidate = useSeriesInvalidator();
  return useMutation({ mutationFn: (exportId: string) => api.renderExport(exportId), onSuccess: () => invalidate(seriesId) });
}

export function useDeleteExport(seriesId: string) {
  const invalidate = useSeriesInvalidator();
  return useMutation({ mutationFn: (exportId: string) => api.deleteExport(exportId), onSuccess: () => invalidate(seriesId) });
}

export function useImportChannel() {
  const invalidate = useSeriesInvalidator();
  return useMutation({
    mutationFn: ({ url, groups }: { url: string; groups: { title: string; series_id: string | null; episodes: unknown[] }[] }) =>
      api.importChannel(url, groups),
    onSuccess: () => invalidate(),
  });
}

export function useAddVideo() {
  const invalidate = useSeriesInvalidator();
  return useMutation({
    mutationFn: ({ url, seriesId, title }: { url: string; seriesId?: string; title?: string }) =>
      api.addVideo(url, seriesId, title),
    onSuccess: () => invalidate(),
  });
}

export function useSaveCookies() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: ({ platform, cookies }: { platform: string; cookies: string }) => api.saveCookies(platform, cookies),
    onSuccess: () => qc.invalidateQueries({ queryKey: keys.providers }),
  });
}

export function useSaveAppSettings() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (settings: AppSettings) => api.saveAppSettings(settings),
    onSuccess: (saved) => {
      qc.setQueryData(keys.appSettings, saved);
      // a saved key/address changes what a service lists
      qc.invalidateQueries({ queryKey: [...keys.appSettings, "models"] });
      qc.invalidateQueries({ queryKey: [...keys.appSettings, "voices"] });
    },
  });
}
