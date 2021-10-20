/*
Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.

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
#ifndef NVIDIA_GXF_CORE_GXF_H_
#define NVIDIA_GXF_CORE_GXF_H_

#ifdef __cplusplus
#include <cstdint>
#include <fstream>
#else
#include <stdint.h>
#endif

#ifdef __cplusplus
extern "C" {
#endif

// --  Result type  --------------------------------------------------------------------------------

/// @brief GXF error and result codes which are used by allmost all GXF functions.
typedef enum {
  GXF_SUCCESS = 0,
  GXF_FAILURE = 1,

  GXF_NOT_IMPLEMENTED,
  GXF_FILE_NOT_FOUND,
  GXF_INVALID_ENUM,
  GXF_NULL_POINTER,
  GXF_UNINITIALIZED_VALUE,

  GXF_ARGUMENT_NULL,
  GXF_ARGUMENT_OUT_OF_RANGE,
  GXF_ARGUMENT_INVALID,

  GXF_OUT_OF_MEMORY,
  GXF_MEMORY_INVALID_STORAGE_MODE,

  GXF_CONTEXT_INVALID,

  GXF_EXTENSION_NOT_FOUND,
  GXF_EXTENSION_FILE_NOT_FOUND,
  GXF_EXTENSION_NO_FACTORY,

  GXF_FACTORY_TOO_MANY_COMPONENTS,
  GXF_FACTORY_DUPLICATE_TID,
  GXF_FACTORY_UNKNOWN_TID,
  GXF_FACTORY_ABSTRACT_CLASS,
  GXF_FACTORY_UNKNOWN_CLASS_NAME,
  GXF_FACTORY_INVALID_INFO,
  GXF_FACTORY_INCOMPATIBLE,

  GXF_ENTITY_NOT_FOUND,
  GXF_ENTITY_COMPONENT_NOT_FOUND,
  GXF_ENTITY_CAN_NOT_ADD_COMPONENT_AFTER_INITIALIZATION,

  GXF_PARAMETER_NOT_FOUND,
  GXF_PARAMETER_ALREADY_REGISTERED,
  GXF_PARAMETER_INVALID_TYPE,
  GXF_PARAMETER_OUT_OF_RANGE,
  GXF_PARAMETER_NOT_INITIALIZED,
  GXF_PARAMETER_CAN_NOT_MODIFY_CONSTANT,
  GXF_PARAMETER_PARSER_ERROR,
  GXF_PARAMETER_NOT_NUMERIC,
  GXF_PARAMETER_MANDATORY_NOT_SET,

  GXF_CONTRACT_INVALID_SEQUENCE,
  GXF_CONTRACT_PARAMETER_NOT_SET,
  GXF_CONTRACT_MESSAGE_NOT_AVAILABLE,

  GXF_INVALID_LIFECYCLE_STAGE,
  GXF_INVALID_EXECUTION_SEQUENCE,
  GXF_REF_COUNT_NEGATIVE,
  GXF_RESULT_ARRAY_TOO_SMALL,

  GXF_INVALID_DATA_FORMAT,

  GXF_EXCEEDING_PREALLOCATED_SIZE,

  GXF_QUERY_NOT_ENOUGH_CAPACITY,
  GXF_QUERY_NOT_APPLICABLE,
  GXF_QUERY_NOT_FOUND,
} gxf_result_t;

/// @brief Gets a string describing an GXF error code.
///
/// The caller does not get ownership of the return C string and must not delete it.
///
/// @param result A GXF error code
/// @return A pointer to a C string with the error code description.
const char* GxfResultStr(gxf_result_t result);

// --  Unique identifiers  -------------------------------------------------------------------------

/// @brief Type of unique GXF object identifiers (UID/uid)
///
/// Uids are used to reference entities and components throughout the GXF API.
typedef int64_t gxf_uid_t;

/// @brief A GXF UID which can be used to indicate an invalid or unused GXF UID.
#define kNullUid 0L
/// @brief A GXF UID which can be used to indicate an unspecified component during
/// graph load operation. This component should be updated in a subsequent
/// graph/parameters file. Failing to do so will result in an error during graph activation
#define kUnspecifiedUid -1L

/// @brief Type of unique GXF type identifier (TID/tid)
///
/// Tids are used to uniquely identify the type of a component, instead of for example using
/// a string with the type name.
typedef struct {
  uint64_t hash1;
  uint64_t hash2;
} gxf_tid_t;

/// @brief Gives the null value of a GXF tid
///
/// The null tid is used to mark a tid as invalid or to indicate that a tid is not set.
///
/// @return The null tid
inline gxf_tid_t GxfTidNull() {
  gxf_tid_t result;
  result.hash1 = 0UL;
  result.hash2 = 0UL;
  return result;
}

/// @brief Determines if a GXF tid is null
///
/// @param tid A GXF tid
/// @return Returns 1 if 'tid' is the null tid; and 0 otherwise.
inline uint32_t GxfTidIsNull(gxf_tid_t tid) {
  return (tid.hash1 == 0UL && tid.hash2 == 0UL) ? 1 : 0;
}

// --  Context  ------------------------------------------------------------------------------------

/// @brief Type for context handle
typedef void* gxf_context_t;

/// @brief An invalid context
#define kNullContext nullptr

/// @brief GXF Core Version
#define kGxfCoreVersion "2.3.1"

/// @brief Creates a new GXF context
///
/// A GXF context is required for all almost all GXF operations. The context must be destroyed with
/// 'GxfContextDestroy'. Multiple contexts can be created in the same process, however they can not
/// communicate with each other.
///
/// @param context The new GXF context is written to the given pointer.
/// @return GXF_SUCCESS if the operation was successful, or otherwise one of the GXF error codes.
gxf_result_t GxfContextCreate(gxf_context_t* context);

/// @brief Destroys a GXF context
///
/// Every GXF context must be destroyed by calling this function. The context must have been
/// previously created with 'GxfContextCreate'. This will also destroy all entities and components
/// which were created as part of the context.
///
/// @param context A valid GXF context.
/// @return GXF_SUCCESS if the operation was successful, or otherwise one of the GXF error codes.
gxf_result_t GxfContextDestroy(gxf_context_t context);

// --  Extensions  ---------------------------------------------------------------------------------

/// @brief Maximum number of extensions in a context
#define kMaxExtensions 1024

/// @deprecated Use 'GxfLoadExtensions' instead
gxf_result_t GxfLoadExtension(gxf_context_t context, const char* filename);

/// @deprecated Use 'GxfLoadExtensions' instead
gxf_result_t GxfLoadExtensionManifest(gxf_context_t context, const char* manifest_filename);

/// @brief Registers a component with a GXF extension
///
/// A GXF extension need to register all of its components in the extension factory function. For
/// convenience the helper macros in gxf/std/extension_factory_helper.hpp can be used.
///
/// The developer must choose a unique GXF tid with two random 64-bit integers. The developer
/// must ensure that every GXF component has a unique tid. The name of the component must be
/// the fully qualified C++ type name of the component. A component may only have a single base
/// class and that base class must be specified with its fully qualified C++ type name as the
/// parameter 'base_name'.
///
/// @see gxf/std/extension_factory_helper.hpp
/// @see core/type_name.hpp
///
/// @param context A valid GXF context
/// @param tid The chosen GXF tid
/// @param name The type name of the component
/// @param base_name The type name of the base class of the component
/// @return GXF_SUCCESS if the operation was successful, or otherwise one of the GXF error codes.
gxf_result_t GxfRegisterComponent(gxf_context_t context, gxf_tid_t tid, const char* name,
                                  const char* base_name);

// --  Entities  -----------------------------------------------------------------------------------

/// @brief Maximum number of entities in a context
#define kMaxEntities 1024

typedef enum {
  GXF_ENTITY_STATUS_NOT_STARTED = 0,
  GXF_ENTITY_STATUS_START_PENDING = 1,
  GXF_ENTITY_STATUS_STARTED = 2,
  GXF_ENTITY_STATUS_TICK_PENDING = 3,
  GXF_ENTITY_STATUS_STOP_PENDING = 4,
} gxf_entity_status_t;

/// @deprecated Use 'GxfCreateEntity' instead
gxf_result_t GxfEntityCreate(gxf_context_t context, gxf_uid_t* eid);

/// @brief Activates a previously created and inactive entity
///
/// Activating an entity generally marks the official start of its lifetime and has multiple
/// implications:
/// - If mandatory parameters, i.e. parameter which do not hav the flag "optional", are not set
///   the operation will fail.
/// - All components on the entity are initialized.
/// - All codelets on the entity are scheduled for execution. The scheduler will start calling
///   start, tick and stop functions as specified by scheduling terms.
/// - After activation trying to change a dynamic parameters will result in a failure.
/// - Adding or removing components of an entity after activation will result in a failure.
///
/// @param context A valid GXF context
/// @param eid UID of a valid entity
/// @return GXF error code
gxf_result_t GxfEntityActivate(gxf_context_t context, gxf_uid_t eid);

/// @brief Deactivates a previously activated entity
///
/// Deactivating an entity generally marks the official end of its lifetime and has multiple
/// implications:
/// - All codelets are removed from the schedule. Already running entities are run to completion.
/// - All components on the entity are deinitialized.
/// - Components can be added or removed again once the entity was deactivated.
/// - Mandatory and non-dynamic parameters can be changed again.
///
/// Note: In case that the entity is currently executing this function will wait and block until
///       the current execution is finished.
///
/// @param context A valid GXF context
/// @param eid UID of a valid entity
/// @return GXF error code
gxf_result_t GxfEntityDeactivate(gxf_context_t context, gxf_uid_t eid);

/// @brief Destroyes a previously created entity
///
/// Destroys an entity immediately. The entity is destroyed even if the reference count has not
/// yet reached 0. If the entity is active it is deactivated first.
///
/// Note: This function can block for the same reasons as 'GxfEntityDeactivate'.
///
/// @param context A valid GXF context
/// @param eid The returned UID of the created entity
/// @return GXF_SUCCESS if the operation was successful, or otherwise one of the GXF error codes.
gxf_result_t GxfEntityDestroy(gxf_context_t context, gxf_uid_t eid);

/// @brief Finds an entity by its name
///
/// @param context A valid GXF context
/// @param name A C string with the name of the entity. Ownership is not transferred.
/// @param eid The returned UID of the entity
/// @return GXF_SUCCESS if the operation was successful, or otherwise one of the GXF error codes.
gxf_result_t GxfEntityFind(gxf_context_t context, const char* name, gxf_uid_t* eid);

/// @brief Finds all entities in the current application
///
/// Finds and returns all entity ids for the current application. If more than `max_entities` exist
/// only `max_entities` will be returned. The order and selection of entities returned is abritrary.
///
/// @param context A valid GXF context
/// @param num_entities In/Out: the max number of entities that can fit in the buffer/the number of
/// entities that exist in the application
/// @param entities A buffer allocated by the caller for returned UIDs of all entities, with
/// capacity for `num_entities`.
/// @return GXF_SUCCESS if the operation was successful, GXF_QUERY_NOT_ENOUGH_CAPACITY if more
/// entites exist in the application than `max_entities`, or otherwise one of the GXF error codes.
gxf_result_t GxfEntityFindAll(gxf_context_t context, uint64_t* num_entities, gxf_uid_t* entities);

/// @brief Increases the reference count for an entity by 1.
///
/// By default reference counting is disabled for an entity. This means that entities created with
/// 'GxfEntityCreate' are not automatically destroyed. If this function is called for an entity
/// with disabled reference count, reference counting is enabled and the reference count is set to
/// 1. Once reference counting is enabled an entity will be automatically destroyed if the reference
/// count reaches zero, or if 'GxfEntityCreate' is called explicitely.
///
/// @param context A valid GXF context
/// @param eid The UID of a valid entity
/// @return GXF_SUCCESS if the operation was successful, or otherwise one of the GXF error codes.
gxf_result_t GxfEntityRefCountInc(gxf_context_t context, gxf_uid_t eid);

/// @brief Decreases the reference count for an entity by 1.
///
/// See 'GxfEntityRefCountInc' for more details on reference counting.
///
/// @param context A valid GXF context
/// @param eid The UID of a valid entity
/// @return GXF_SUCCESS if the operation was successful, or otherwise one of the GXF error codes.
gxf_result_t GxfEntityRefCountDec(gxf_context_t context, gxf_uid_t eid);

/// @brief Gets the status of the entity.
///
/// See 'gxf_entity_status_t' for the various status.
///
/// @param context A valid GXF context
/// @param eid The UID of a valid entity
/// @param entity_status output; status of an entity eid
/// @return GXF_SUCCESS if the operation was successful, or otherwise one of the GXF error codes.
gxf_result_t GxfEntityGetStatus(gxf_context_t context, gxf_uid_t eid,
                              gxf_entity_status_t* entity_status);

/// @brief Notifies the occurrence of an event and inform the scheduler to check the status of the
/// entity
///
/// The entity must have an 'AsynchronousSchedulingTerm' scheduling term component and it must be in
/// "EVENT_WAITING" state for the notification to be acknowledged.
///
///  See 'AsynchronousEventState' for various states
///
/// @param context A valid GXF context
/// @param eid The UID of a valid entity
/// @return GXF_SUCCESS if the operation was successful, or otherwise one of the GXF error codes.

gxf_result_t GxfEntityEventNotify(gxf_context_t context, gxf_uid_t eid);

// --  Components  ---------------------------------------------------------------------------------

/// @brief Maximum number of components in an entity or extension
#define kMaxComponents 1024

/// @brief Gets the GXF unique type ID (TID) of a component
///
/// Get the unique type ID which was used to register the component with GXF. The function expects
/// the fully qualified C++ type name of the component including namespaces.
///
/// Example of a valid component type name: "nvidia::gxf::test::PingTx"
///
/// @param context A valid GXF context
/// @param name The fully qualified C++ type name of the component
/// @param tid The returned TID of the component
/// @return GXF_SUCCESS if the operation was successful, or otherwise one of the GXF error codes.
gxf_result_t GxfComponentTypeId(gxf_context_t context, const char* name, gxf_tid_t* tid);

/// @brief Gets the fully qualified C++ type name GXF component typename
///
/// Get the unique typename of the component with which it was registered using one of
/// the GXF_EXT_FACTORY_ADD*() macros
///
/// @param context A valid GXF context
/// @param tid The unique type ID (TID) of the component with which the component was registered
/// @param name The returned name of the component
/// @return GXF_SUCCESS if the operation was successful, or otherwise one of the GXF error codes.
gxf_result_t GxfComponentTypeName(gxf_context_t context, gxf_tid_t tid, const char** name);

/// @brief Gets the name of a component
///
/// Each component has a user-defined name which was used in the call to 'GxfComponentAdd'.
/// Usually the name is specified in the GXF application file.
///
/// @param context A valid GXF context
/// @param cid The unique object ID (UID) of the component
/// @param name The returned name of the component
/// @return GXF_SUCCESS if the operation was successful, or otherwise one of the GXF error codes.
gxf_result_t GxfComponentName(gxf_context_t context, gxf_uid_t cid, const char** name);

/// @brief Gets the unique object ID of the entity of a component
///
/// Each component has a unique ID with respect to the context and is stored in one entity. This
/// function can be used to retreive the ID of the entity to which a given component belongs.
///
/// @param context A valid GXF context
/// @param cid The unique object ID (UID) of the component
/// @param eid The returned UID of the entity
/// @return GXF_SUCCESS if the operation was successful, or otherwise one of the GXF error codes.
gxf_result_t GxfComponentEntity(gxf_context_t context, gxf_uid_t cid, gxf_uid_t* eid);

/// @brief Adds a new component to an entity
///
/// An entity can contain multiple components and this function can be used to add a new component
/// to an entity. A component must be added before an entity is activated, or after it was
/// deactivated. Components must not be added to active entities. The order of components is stable
/// and identical to the order in which components are added (see 'GxfComponentFind').
///
/// @param context A valid GXF context
/// @param eid The unique object ID (UID) of the entity to which the component is added.
/// @param tid The unique type ID (TID) of the component to be added to the entity.
/// @param name The name of the new component. Ownership is not transferred.
/// @param cid The returned UID of the created component
/// @return GXF_SUCCESS if the operation was successful, or otherwise one of the GXF error codes.
gxf_result_t GxfComponentAdd(gxf_context_t context, gxf_uid_t eid, gxf_tid_t tid, const char* name,
                             gxf_uid_t* cid);

/// @brief Adds an existing component to the interface of an entity
///
/// An entity can holds references to other components in its interface, so that when finding a
/// component in an entity, both the component this entity holds and those it refers to will be
/// returned.
/// This supports the case when an entity contains a subgraph, then those components that has been
/// declared in the subgraph interface will be put to the interface of the parent entity.
///
/// @param context A valid GXF context
/// @param eid The unique object ID (UID) of the entity to which the component is added.
/// @param cid The unique object ID of the component
/// @param name The name of the new component. Ownership is not transferred.
/// @return GXF_SUCCESS if the operation was successful, or otherwise one of the GXF error codes.
gxf_result_t GxfComponentAddToInterface(gxf_context_t context, gxf_uid_t eid,
                                        gxf_uid_t cid, const char* name);

/// @brief Finds a component in an entity
///
/// Searches components in an entity which satisfy certain criteria: component type, component
/// name, and component min index. All three criteria are optional; in case no criteria is given
/// the first component is returned. The main use case for "component min index" is a repeated
/// search which continues at the index which was returned by a previous search.
///
/// In case no entity with the given criteria was found GXF_ENTITY_NOT_FOUND is returned.
///
/// @param context A valid GXF context
/// @param eid The unique object ID (UID) of the entity which is searched.
/// @param tid The component type ID (TID) of the component to find (optional)
/// @param name The component name of the component to find (optional). Ownership not transferred.
/// @param offset The index of the first component in the entity to search. Also contains the index
///               of the component which was found.
/// @param cid The returned UID of the searched component
/// @return GXF_SUCCESS if a component matching the criteria was found, GXF_ENTITY_NOT_FOUND if no
///         component matching the criteria was found, or otherwise one of the GXF error codes.
gxf_result_t GxfComponentFind(gxf_context_t context, gxf_uid_t eid, gxf_tid_t tid, const char* name,
                              int32_t* offset, gxf_uid_t* cid);

/// @brief Gets the component type ID (TID) of a component
///
/// @param cid The component object ID (UID) for which the component type is requested.
/// @param tid The returned TID of the component
/// @return GXF_SUCCESS if the operation was successful, or otherwise one of the GXF error codes.
gxf_result_t GxfComponentType(gxf_context_t context, gxf_uid_t cid, gxf_tid_t* tid);

/// @brief Verifies that a component exists, has the given type, gets a pointer to it.
///
/// @param tid The expected component type ID (TID) of the component
/// @param pointer The returned pointer to the component object.
/// @return GXF_SUCCESS if the operation was successful, or otherwise one of the GXF error codes.
gxf_result_t GxfComponentPointer(gxf_context_t context, gxf_uid_t uid, gxf_tid_t tid,
                                 void** pointer);

// --  Parameter  ----------------------------------------------------------------------------------

/// @brief [experimental] The name of the parameter which stores the name of a component
#define kInternalNameParameterKey "__name"

/// @brief Flags describing the behavior of parameter
///
/// Parameter flags are specified when a parameter is registered as part of the component interface.
/// Multipl flags can be OR combined.
enum gxf_parameter_flags_t_ {
  /// No additional flags are set (the default). This means the parameter is mandatory and static.
  /// The parameter must be set before entity activation and can not be changed after entity
  /// activation.
  GXF_PARAMETER_FLAGS_NONE = 0,
  /// The parameter value is optional and might not be available after entity activation.
  /// This implies that it is not allowed to access the parameter with 'get()' in the C++ API.
  /// Instead 'try_get' must be used.
  GXF_PARAMETER_FLAGS_OPTIONAL = 1,
  /// The parameter is dynamic an might change after entity activation. However it is still
  /// guaranteed that parameters do not change during the initialize, deinitialize, start, tick,
  /// or stop functions.
  GXF_PARAMETER_FLAGS_DYNAMIC = 2
};

/// @brief Type used for parameter flags.
///
/// @see gxf_parameter_flags_t_
typedef uint32_t gxf_parameter_flags_t;

// Sets a 64-bit floating point parameter
gxf_result_t GxfParameterSetFloat64(gxf_context_t context, gxf_uid_t uid, const char* key,
                                    double value);
// Sets a 64-bit signed integer parameter
gxf_result_t GxfParameterSetInt64(gxf_context_t context, gxf_uid_t uid, const char* key,
                                  int64_t value);
// Sets a 64-bit unsigned integer parameter
gxf_result_t GxfParameterSetUInt64(gxf_context_t context, gxf_uid_t uid, const char* key,
                                   uint64_t value);
// Sets a 32-bit unsigned integer parameter
gxf_result_t GxfParameterSetUInt32(gxf_context_t context, gxf_uid_t uid, const char* key,
                                   uint32_t value);
// Sets a 16-bit unsigned integer parameter
gxf_result_t GxfParameterSetUInt16(gxf_context_t context, gxf_uid_t uid, const char* key,
                                   uint16_t value);
// Sets a string parameter
gxf_result_t GxfParameterSetStr(gxf_context_t context, gxf_uid_t uid, const char* key,
                                const char* value);
// Sets a handle parameter
gxf_result_t GxfParameterSetHandle(gxf_context_t context, gxf_uid_t uid, const char* key,
                                   gxf_uid_t cid);
// Sets a boolean parameter
gxf_result_t GxfParameterSetBool(gxf_context_t context, gxf_uid_t uid, const char* key,
                                 bool value);
// Sets a 32-bit signed integer parameter
gxf_result_t GxfParameterSetInt32(gxf_context_t context, gxf_uid_t uid, const char* key,
                                  int32_t value);
// Sets a String 1D-Vector parameter. Internally the data is stored in
// a std::vector. The length of the vector should match the length of the registered parameter.
gxf_result_t GxfParameterSet1DStrVector(gxf_context_t context, gxf_uid_t uid, const char* key,
                                        const char* value[], uint64_t length);
// Sets a Float 64 1D-Vector parameter. Internally the data is stored in
// a std::vector. The length of the vector should match the length of the registered parameter.
gxf_result_t GxfParameterSet1DFloat64Vector(gxf_context_t context, gxf_uid_t uid, const char* key,
                                            double* value, uint64_t length);
// Sets a Float 64 2D-Vector parameter. Internally the data is stored in a
// std::vector<std::vector<TYPE>. The height should match the length of the outer std::vector
// and the width should match the length of all the inner std::vectors. Also, these height and
// width should match the shape of the registered parameters.
gxf_result_t GxfParameterSet2DFloat64Vector(gxf_context_t context, gxf_uid_t uid, const char* key,
                                            double** value, uint64_t height, uint64_t width);
// Sets a signed 64-bit int 1D-Vector parameter. Internally the data is stored in
// a std::vector. The length of the vector should match the length of the registered parameter.
gxf_result_t GxfParameterSet1DInt64Vector(gxf_context_t context, gxf_uid_t uid, const char* key,
                                          int64_t* value, uint64_t length);
// Sets a signed 64-bit int 2D-Vector parameter. Internally the data is stored in a
// std::vector<std::vector<TYPE>. The height should match the length of the outer std::vector
// and the width should match the length of all the inner std::vectors. Also, these height and
// width should match the shape of the registered parameters.
gxf_result_t GxfParameterSet2DInt64Vector(gxf_context_t context, gxf_uid_t uid, const char* key,
                                          int64_t** value, uint64_t height, uint64_t width);
// Sets a unsigned 64-bit unsigned int 1D-Vector parameter. Internally the data is stored in
// a std::vector. The length of the vector should match the length of the registered parameter.
gxf_result_t GxfParameterSet1DUInt64Vector(gxf_context_t context, gxf_uid_t uid, const char* key,
                                           uint64_t* value, uint64_t length);
// Sets a unsigned 64-bit unsigned int 2D-Vector parameter. Internally the data is stored in a
// std::vector<std::vector<TYPE>. The height should match the length of the outer std::vector
// and the width should match the length of all the inner std::vectors. Also, these height and
// width should match the shape of the registered parameters.
gxf_result_t GxfParameterSet2DUInt64Vector(gxf_context_t context, gxf_uid_t uid, const char* key,
                                           uint64_t** value, uint64_t height, uint64_t width);
// Sets a signed 32-bit int 1-D Vector parameter. Internally the data is stored in
// a std::vector. The length of the vector should match the length of the registered parameter.
gxf_result_t GxfParameterSet1DInt32Vector(gxf_context_t context, gxf_uid_t uid, const char* key,
                                          int32_t* value, uint64_t length);
// Sets a signed 32-bit int 2-D Vector parameter. Internally the data is stored in a
// std::vector<std::vector<TYPE>. The height should match the length of the outer std::vector
// and the width should match the length of all the inner std::vectors. Also, these height and
// width should match the shape of the registered parameters.
gxf_result_t GxfParameterSet2DInt32Vector(gxf_context_t context, gxf_uid_t uid, const char* key,
                                          int32_t** value, uint64_t height, uint64_t width);

// Gets a 64-bit floating point parameter
gxf_result_t GxfParameterGetFloat64(gxf_context_t context, gxf_uid_t uid, const char* key,
                                    double* value);
// Gets a 64-bit signed integer parameter
gxf_result_t GxfParameterGetInt64(gxf_context_t context, gxf_uid_t uid, const char* key,
                                  int64_t* value);
// Gets a 64-bit unsigned integer parameter
gxf_result_t GxfParameterGetUInt64(gxf_context_t context, gxf_uid_t uid, const char* key,
                                   uint64_t* value);
// Gets a 32-bit unsigned integer parameter
gxf_result_t GxfParameterGetUInt32(gxf_context_t context, gxf_uid_t uid, const char* key,
                                   uint32_t* value);
// Gets a 16-bit unsigned integer parameter
gxf_result_t GxfParameterGetUInt16(gxf_context_t context, gxf_uid_t uid, const char* key,
                                   uint16_t* value);
// Gets a string parameter
gxf_result_t GxfParameterGetStr(gxf_context_t context, gxf_uid_t uid, const char* key,
                                const char** value);
// Gets a file path parameter
gxf_result_t GxfParameterGetPath(gxf_context_t context, gxf_uid_t uid, const char* key,
                                const char** value);
// Gets a handle parameter
gxf_result_t GxfParameterGetHandle(gxf_context_t context, gxf_uid_t uid, const char* key,
                                   gxf_uid_t* cid);
// Gets a bool parameter
gxf_result_t GxfParameterGetBool(gxf_context_t context, gxf_uid_t uid, const char* key,
                                 bool* value);
// Gets a 32-bit signed integer parameter
gxf_result_t GxfParameterGetInt32(gxf_context_t context, gxf_uid_t uid, const char* key,
                                  int32_t* value);
// Gets the length of the 1D vector.
// For 1D: rank = 1; shape = [length, ...]
gxf_result_t GxfParameterGet1DFloat64VectorInfo(gxf_context_t context, gxf_uid_t uid,
                                                const char* key, uint64_t* length);
// Gets the height/width of the 2D vector.
// For 2D: rank = 2; shape = [height, width, ...]
gxf_result_t GxfParameterGet2DFloat64VectorInfo(gxf_context_t context, gxf_uid_t uid,
                                                const char* key, uint64_t* height, uint64_t* width);
// Gets the length of the 1D vector.
// For 1D: rank = 1; shape = [length, ...]
gxf_result_t GxfParameterGet1DInt64VectorInfo(gxf_context_t context, gxf_uid_t uid, const char* key,
                                              uint64_t* length);
// Gets the height/width of the 2D vector.
// For 2D: rank = 2; shape = [height, width, ...]
gxf_result_t GxfParameterGet2DInt64VectorInfo(gxf_context_t context, gxf_uid_t uid, const char* key,
                                              uint64_t* height, uint64_t* width);
// Gets the length of the 1D vector.
// For 1D: rank = 1; shape = [length, ...]
gxf_result_t GxfParameterGet1DUInt64VectorInfo(gxf_context_t context, gxf_uid_t uid,
                                               const char* key, uint64_t* length);
// Gets the height/width of the 2D vector.
// For 2D: rank = 2; shape = [height, width, ...]
gxf_result_t GxfParameterGet2DUInt64VectorInfo(gxf_context_t context, gxf_uid_t uid,
                                               const char* key, uint64_t* height, uint64_t* width);
// Gets the length of the 1D vector.
// For 1D: rank = 1; shape = [length, ...]
gxf_result_t GxfParameterGet1DInt32VectorInfo(gxf_context_t context, gxf_uid_t uid, const char* key,
                                              uint64_t* length);
// Gets the height/width of the 2D vector.
// For 2D: rank = 2; shape = [height, width, ...]
gxf_result_t GxfParameterGet2DInt32VectorInfo(gxf_context_t context, gxf_uid_t uid, const char* key,
                                              uint64_t* height, uint64_t* width);
// Gets a 1D-Vector of 64-bit floating point.
// For 1D: rank = 1; shape = [length, ...]
// The rank and shape should match with the internal registered std::vector parameter.
// The length can be queried using GxfGetParameterInfo()
gxf_result_t GxfParameterGet1DFloat64Vector(gxf_context_t context, gxf_uid_t uid, const char* key,
                                            double* value, uint64_t* length);
// Gets a 1D-Vector of strings.
// The length can be queried using GxfGetParameterInfo()
gxf_result_t GxfParameterGet1DStrVector(gxf_context_t context, gxf_uid_t uid, const char* key,
                                            char* value[], uint64_t* count, uint64_t* min_length);
// Gets a 2D-Vector of 64-bit floating point.
// For 2D: rank = 2; shape = [height, width, ...]
// The rank and shape should match with the internal registered std::vector parameter.
// The height and width can be obtained using GxfGetParameterInfo()
gxf_result_t GxfParameterGet2DFloat64Vector(gxf_context_t context, gxf_uid_t uid, const char* key,
                                            double** value, uint64_t* height, uint64_t* width);
// Gets a 1D-Vector of signed 64-bit integers.
// For 1D: rank = 1; shape = [length, ...]
// The rank and shape should match with the internal registered std::vector parameter.
// The length can be queried using GxfGetParameterInfo()
gxf_result_t GxfParameterGet1DInt64Vector(gxf_context_t context, gxf_uid_t uid, const char* key,
                                          int64_t* value, uint64_t* length);
// Gets a 2D-Vector of signed 64-bit integers.
// For 2D: rank = 2; shape = [height, width, ...]
// The rank and shape should match with the internal registered std::vector parameter.
// The height and width can be obtained using GxfGetParameterInfo()
gxf_result_t GxfParameterGet2DInt64Vector(gxf_context_t context, gxf_uid_t uid, const char* key,
                                          int64_t** value, uint64_t* height, uint64_t* width);
// Gets a 1D-Vector of unsigned 32-bit integers.
// For 1D: rank = 1; shape = [length, ...]
// The rank and shape should match with the internal registered std::vector parameter.
// The length can be queried using GxfGetParameterInfo()
gxf_result_t GxfParameterGet1DUInt64Vector(gxf_context_t context, gxf_uid_t uid, const char* key,
                                           uint64_t* value, uint64_t* length);
// Gets a 2D-Vector of unsigned 32-bit integers.
// For 2D: rank = 2; shape = [height, width, ...]
// The rank and shape should match with the internal registered std::vector parameter.
// The height and width can be obtained using GxfGetParameterInfo()
gxf_result_t GxfParameterGet2DUInt64Vector(gxf_context_t context, gxf_uid_t uid, const char* key,
                                           uint64_t** value, uint64_t* height, uint64_t* width);
// Gets a 1D-Vector of signed 32-bit integers.
// For 1D: rank = 1; shape = [length, ...]
// The rank and shape should match with the internal registered std::vector parameter.
// The length can be queried using GxfGetParameterInfo()
gxf_result_t GxfParameterGet1DInt32Vector(gxf_context_t context, gxf_uid_t uid, const char* key,
                                          int32_t* value, uint64_t* length);
// Gets a 2D-Vector of signed 32-bit integers.
// For 2D: rank = 2; shape = [height, width, ...]
// The rank and shape should match with the internal registered std::vector parameter.
// The height and width can be obtained using GxfGetParameterInfo()
gxf_result_t GxfParameterGet2DInt32Vector(gxf_context_t context, gxf_uid_t uid, const char* key,
                                          int32_t** value, uint64_t* height, uint64_t* width);
// --  Graph execution  ----------------------------------------------------------------------------

// Loads a list of entities from a YAML file.
gxf_result_t GxfGraphLoadFile(gxf_context_t context, const char* filename);

// Loads a list of entities from a YAML file.
gxf_result_t GxfGraphParseString(gxf_context_t context, const char* text);

// Activate all System components
gxf_result_t GxfGraphActivate(gxf_context_t context);

// Deactivate all System components
gxf_result_t GxfGraphDeactivate(gxf_context_t context);

// Starts the execution of the graph asynchronously
gxf_result_t GxfGraphRunAsync(gxf_context_t context);

// Interrupt the execution of the graph
gxf_result_t GxfGraphInterrupt(gxf_context_t context);

// Waits for the graph to complete execution
gxf_result_t GxfGraphWait(gxf_context_t context);

// Runs all System components and waits for their completion
gxf_result_t GxfGraphRun(gxf_context_t context);

// --  Info queries  -------------------------------------------------------------------------------

// Type to represent version of GXF Runtime and list of loaded Extensions
typedef struct {
  const char* version;      // GXF Runtime Version
  uint64_t num_extensions;  // in-out capacity of extensions/Number of extension types
  gxf_tid_t* extensions;    // List of Extension IDs
} gxf_runtime_info;

// Gets Meta Data about the GXF Runtime
gxf_result_t GxfRuntimeInfo(gxf_context_t context, gxf_runtime_info* info);

// Type to represent description and list of components for loaded Extension
typedef struct {
  gxf_tid_t id;             // (UUID) Extension ID (registered via GXF_EXT_FACTORY_SET_INFO)
  const char* name;         // (String) Extension Name (registered via GXF_EXT_FACTORY_SET_INFO)
  const char* description;  // (String) Description (registered via GXF_EXT_FACTORY_SET_INFO)
  const char* version;      // (String) Extension Version (registered via GXF_EXT_FACTORY_SET_INFO)
  const char* runtime_version;
                            // (String) GXF Core version with which the extension was compiled with
  const char* license;      // (String) Extension License (registered via GXF_EXT_FACTORY_SET_INFO)
  const char* author;       // (String) Extension Author (registered via GXF_EXT_FACTORY_SET_INFO)
  uint64_t num_components;  // in-out capacity of components/Number of components
  gxf_tid_t* components;    // List of IDs of provided components
} gxf_extension_info_t;

// Gets description and list of components in loaded Extension
gxf_result_t GxfExtensionInfo(gxf_context_t context, gxf_tid_t tid, gxf_extension_info_t* info);

// Type to represent description and list of Parameter of Component
typedef struct {
  gxf_tid_t cid;            // (UUID) Component ID (registered via GXF_EXT_FACTORY_ADD)
  const char* base_name;    // (String) Base Class Name (registered via GXF_EXT_FACTORY_ADD)
  int is_abstract;          // (Bool) If the Component is abstract (Can not be instantiated)
  const char* type_name;    // (String) Component Name (registered via GXF_EXT_FACTORY_ADD)
  const char* description;  // (String) Description (registered via GXF_EXT_FACTORY_ADD)
  uint64_t num_parameters;  // in-out capacity of parameters/Number of Parameters
  const char** parameters;  // List of Names for Parameters
} gxf_component_info_t;

// Gets description and list of parameters of Component. Parameters are only available after
// at least one instance is created for the Component.
gxf_result_t GxfComponentInfo(gxf_context_t context, gxf_tid_t tid, gxf_component_info_t* info);

/// @brief The type of a parameter
typedef enum {
  GXF_PARAMETER_TYPE_CUSTOM = 0,   // A custom type not natively supported by GXF.
  GXF_PARAMETER_TYPE_HANDLE = 1,   // A GXF handle. The handle type is specified separately.
  GXF_PARAMETER_TYPE_STRING = 2,   // A null-terminated charcter string (const char*)
  GXF_PARAMETER_TYPE_INT64 = 3,    // A 64-bit signed integer (int64_t)
  GXF_PARAMETER_TYPE_UINT64 = 4,   // A 64-bit unsigned integer (uint64_t)
  GXF_PARAMETER_TYPE_FLOAT64 = 5,  // A 64-bit floating point (double)
  GXF_PARAMETER_TYPE_BOOL = 6,     // A boolean type (bool)
  GXF_PARAMETER_TYPE_INT32 = 7,    // A 32-bit signed integer (int32_t)
  GXF_PARAMETER_TYPE_FILE = 8,    // A 32-bit signed integer (int32_t)
} gxf_parameter_type_t;

// Holds metadata information about a parameter which was registered as part of the component
// interface.
typedef struct {
  const char* key;              // The name of the parameter as it appears in the GXF file.
  const char* headline;         // A short headline used to display the parameter to a human.
  const char* description;      // A longer text describing the usage of the parameter.
  gxf_parameter_flags_t flags;  // Parameter flags for example to make a parameter optional.
  gxf_parameter_type_t type;    // The type of the parameter
  gxf_tid_t handle_tid;         // In case the parameter is a handle the TID of the component.
  const void* default_value;    // Default value of parameter, N/A for handle and custom params
  const void* numeric_min;      // Min value of range for numeric parameters, N/A for other types
  const void* numeric_max;      // Max value of range for numeric parameters, N/A for other types
  const void* numeric_step;     // Step value of range for numeric parameters, N/A for other types
  const char* platform_information;
                                // (String) Platforms separated by comma. Empty means all platforms
  int32_t rank;                 // Rank of the parameter. 0-scalar, 1-list etc. Max rank is 8
  int32_t shape[8];             // Sizes of multi dimensional parameters if it is of fixed
                                // length(array). Shape of a dimension is -1 for scalar and
                                // variable length arrays(vector)
} gxf_parameter_info_t;

// Gives a string describing the parameter type
const char* GxfParameterTypeStr(gxf_parameter_type_t param_type);

// Gives a string describing the flag type
const char* GxfParameterFlagTypeStr(gxf_parameter_flags_t_ flag_type);

// Gets description of specific parameter. Fail if the component is not instantiated yet.
gxf_result_t GxfParameterInfo(gxf_context_t context, gxf_tid_t cid, const char* key,
                              gxf_parameter_info_t* info);

gxf_result_t GxfGetParameterInfo(gxf_context_t context, gxf_tid_t cid, const char* key,
                              gxf_parameter_info_t* info);

gxf_result_t GxfRedirectLog(gxf_context_t context, FILE* fp);

// -------------------------------------------------------------------------------------------------

#ifdef __cplusplus
}  // extern "C"
#endif

#ifdef __cplusplus

inline bool operator==(const gxf_tid_t& lhs, const gxf_tid_t& rhs) noexcept {
  return lhs.hash1 == rhs.hash1 && lhs.hash2 == rhs.hash2;
}

inline bool operator!=(const gxf_tid_t& lhs, const gxf_tid_t& rhs) noexcept {
  return lhs.hash1 != rhs.hash1 || lhs.hash2 != rhs.hash2;
}

inline bool operator<(const gxf_tid_t& lhs, const gxf_tid_t& rhs) noexcept {
  return lhs.hash1 < rhs.hash1 || (lhs.hash1 == rhs.hash1 && lhs.hash2 < rhs.hash2);
}

#endif

#endif  // NVIDIA_GXF_CORE_GXF_H_

// Include the extended API functions.
#include "gxf_ext.h"
